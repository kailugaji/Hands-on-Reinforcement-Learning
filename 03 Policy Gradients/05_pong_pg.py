#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# Policy gradient methods on Pong
# 与前述算法的不同之处：
# The baseline is estimated with a moving average for 1M past transitions, instead of all examples (移动平均)
# Several concurrent environments are used (多环境)
# Gradients are clipped to improve training stability (梯度裁剪)
# https://www.cnblogs.com/kailugaji/
import gym
import ptan
import numpy as np
import argparse
import collections
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim

from lib import common

GAMMA = 0.99 # 折扣率
LEARNING_RATE = 0.01 # 学习率
ENTROPY_BETA = 0.01 # 熵正则化因子 the scale of the entropy bonus
BATCH_SIZE = 128 # 一批xx个样本

REWARD_STEPS = 10
# 用于说明一条记录中包含的步(step)数 (sub-trajectories of length 10)
# how many steps ahead the Bellman equation is unrolled to estimate the discounted total reward of every transition.
BASELINE_STEPS = 1000000 # 移动平均长度
GRAD_L2_CLIP = 0.1 # 梯度裁剪阈值

ENV_COUNT = 32 # 32个并行环境


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))

# To make moving average calculations faster, a deque-backed buffer is created:
# 移动平均
# 假设一个序列为“4、5、8、9、10”，使用的移动平均长度为 3。
# 前两个移动平均值是缺失的。第三个移动平均值是 4、5 和 8 的平均值，第四个值是 5、8 和 9 的平均值，第五个值是 8、9 和 10 的平均值。
class MeanBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.deque = collections.deque(maxlen=capacity)
        # 双端队列（double-ended queue）的缩写，由于两端都能编辑，deque既可以用来实现栈（stack）也可以用来实现队列（queue）。
        self.sum = 0.0

    def add(self, val):
        if len(self.deque) == self.capacity:
            self.sum -= self.deque[0]
        self.deque.append(val)
        self.sum += val

    def mean(self):
        if not self.deque:
            return 0.0
        return self.sum / len(self.deque)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", '--name', required=True, help="Name of the run")
    args = parser.parse_args(['-n', '1'])
    device = torch.device("cuda" if args.cuda else "cpu")

    envs = [make_env() for _ in range(ENV_COUNT)]
    writer = SummaryWriter(comment="-pong-pg-" + args.name)

    net = common.AtariPGN(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    # 三卷积，两全连接
    # print(net)
    # AtariPGN(
    #   (conv): Sequential(
    #     (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
    #     (1): ReLU()
    #     (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    #     (3): ReLU()
    #     (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    #     (5): ReLU()
    #   )
    #   (fc): Sequential(
    #     (0): Linear(in_features=3136, out_features=512, bias=True)
    #     (1): ReLU()
    #     (2): Linear(in_features=512, out_features=6, bias=True)
    #   )
    # )

    agent = ptan.agent.PolicyAgent(net, apply_softmax=True, device=device)
    # PolicyAgent：连续
    # make a decision about actions for every observation (依概率)
    # apply_softmax=True：网络输出先经过softmax转化成概率，再从这个概率分布中进行随机抽样
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    # 返回运行记录以用于训练模型，输出格式为：(state, action, reward, last_state)
    # 并不会输出每一步的信息，而是把多步的交互结果综合(累计多步的reward;显示头尾的状态)到一条Experience输出
    # 多步rewards的累加是有衰退的，而其中的衰退系数由参数gamma(折扣率)指定，即reward=r1+gamma∗r2+(gamma^2)∗r3
    # 其中rn代表第n步操作获得的reward
    # last_state: the state we've got after executing the action. If our episode ends, we have None here
    # steps_count=REWARD_STEPS：unroll the Bellman equation for 10 steps
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3) # Adam优化
    # eps：终止条件

    total_rewards = [] # 前几局的奖励之和
    step_idx = 0 # 迭代次数/轮数
    done_episodes = 0 # 局数，几局游戏
    train_step_idx = 0
    baseline_buf = MeanBuffer(BASELINE_STEPS)

    batch_states, batch_actions, batch_scales = [], [], []
    m_baseline, m_batch_scales, m_loss_entropy, m_loss_policy, m_loss_total = [], [], [], [], []
    m_grad_max, m_grad_mean = [], []
    sum_reward = 0.0

    with common.RewardTracker(writer, stop_reward=18) as tracker:
        for step_idx, exp in enumerate(exp_source): # (state, action, reward, last_state)
            baseline_buf.add(exp.reward) # the sum of the discounted reward for every transition
            baseline = baseline_buf.mean() # the discounted reward的移动平均
            batch_states.append(np.array(exp.state, copy=False))
            batch_actions.append(int(exp.action))
            batch_scales.append(exp.reward - baseline) # 优势函数，引入基准(折扣奖励的移动平均)

            # handle new rewards
            new_rewards = exp_source.pop_total_rewards() # 返回一局游戏过后的total_rewords
            if new_rewards:
                if tracker.reward(new_rewards[0], step_idx):
                    break

            if len(batch_states) < BATCH_SIZE: # state里面还没超过BATCH_SIZE个样本
                continue

            train_step_idx += 1
            states_v = torch.FloatTensor(np.array(batch_states, copy=False)).to(device)
            batch_actions_t = torch.LongTensor(batch_actions).to(device)

            scale_std = np.std(batch_scales)
            batch_scale_v = torch.FloatTensor(batch_scales).to(device)

            optimizer.zero_grad()
            logits_v = net(states_v) # 输入状态，输出Q(s, a)值
            log_prob_v = F.log_softmax(logits_v, dim=1) # 输出动作的对数概率分布 log π(s, a)
            log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
            # max (the discounted reward - baseline) * log π(s, a)
            loss_policy_v = -log_prob_actions_v.mean() # min

            # add the entropy bonus to the loss
            prob_v = F.softmax(logits_v, dim=1) # 输出动作的概率分布π(s, a)
            entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean() # 熵正则项
            # 信息熵的计算公式：entropy = -sum (π(s, a) * log π(s, a))
            entropy_loss_v = -ENTROPY_BETA * entropy_v # loss = - beta * entropy
            loss_v = loss_policy_v + entropy_loss_v
            # min -(the discounted reward - baseline) * log π(s, a) + beta * sum (π(s, a) * log π(s, a))

            loss_v.backward()
            nn_utils.clip_grad_norm_(net.parameters(), GRAD_L2_CLIP)
            # 梯度裁剪 类似于PPO https://www.cnblogs.com/kailugaji/p/15396437.html#_lab2_0_1
            # clip(parameters, 1-GRAD_L2_CLIP, 1+GRAD_L2_CLIP)
            optimizer.step()

            # calculate the Kullback-Leibler (KL) divergence between the new policy and the old policy
            new_logits_v = net(states_v) # 输入状态，输出Q(s, a)值
            new_prob_v = F.softmax(new_logits_v, dim=1) # 输出动作的概率分布π(s, a)
            kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
            # KL散度的计算公式：KL = - sum(log(π'(s, a)/π(s, a)) * π(s, a))
            writer.add_scalar("kl", kl_div_v.item(), step_idx)

            # calculate the statistics about the gradients on this training step
            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in net.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item()) # the graph of the maximum
                grad_means += (p.grad ** 2).mean().sqrt().item() # L2 norm of gradients
                grad_count += 1

            writer.add_scalar("baseline", baseline, step_idx)
            writer.add_scalar("entropy", entropy_v.item(), step_idx)
            writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
            writer.add_scalar("batch_scales_std", scale_std, step_idx)
            writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
            writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
            writer.add_scalar("loss_total", loss_v.item(), step_idx)
            writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
            writer.add_scalar("grad_max", grad_max, step_idx)

            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()

    writer.close()
