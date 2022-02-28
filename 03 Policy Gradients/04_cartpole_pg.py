#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# Policy gradient methods on CartPole
# 为增加探索，引入entropy bonus(信息熵正则项)
# 为了鼓励模型加入更多的不确定性，这样在训练的时候，模型就会去探索更多的可能性
# H(π) = - sum(π(a|s) * log π(a|s))
# min -(the discounted reward - baseline) * log π(s, a) + beta * sum (π(s, a) * log π(s, a))
# https://www.cnblogs.com/kailugaji/
import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99 # 折扣率
LEARNING_RATE = 0.01 # 学习率
ENTROPY_BETA = 0.01 # 熵正则化因子 the scale of the entropy bonus
BATCH_SIZE = 16 # 一批xx个样本
REWARD_STEPS = 10
# 用于说明一条记录中包含的步(step)数 (sub-trajectories of length 10)
# how many steps ahead the Bellman equation is unrolled to estimate the discounted total reward of every transition.

# 构建网络
class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        # input_size：输入状态维度，hidden_size：隐层维度=128，n_actions：输出动作维度
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128), # 全连接层，隐层预设为128维度
            nn.ReLU(),
            nn.Linear(128, n_actions) # 全连接层
        )

    def forward(self, x):
        return self.net(x) # 未用softmax

# 平滑
def smooth(old: Optional[float], val: float, alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val


if __name__ == "__main__":
    env = gym.make("CartPole-v0") # 创建游戏环境
    writer = SummaryWriter(comment="-cartpole-pg")

    net = PGN(env.observation_space.shape[0], env.action_space.n) # 4(状态)->128->2(动作)
    # print(net)

    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)
    # PolicyAgent：连续
    # make a decision about actions for every observation (依概率)
    # apply_softmax=True：网络输出先经过softmax转化成概率，再从这个概率分布中进行随机抽样
    # float32_preprocessor：returns the observation as float64 instead of the float32 required by PyTorch
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    # 返回运行记录以用于训练模型，输出格式为：(state, action, reward, last_state)
    # 并不会输出每一步的信息，而是把多步的交互结果综合(累计多步的reward;显示头尾的状态)到一条Experience输出
    # 多步rewards的累加是有衰退的，而其中的衰退系数由参数gamma(折扣率)指定，即reward=r1+gamma∗r2+(gamma^2)∗r3
    # 其中rn代表第n步操作获得的reward
    # last_state: the state we've got after executing the action. If our episode ends, we have None here
    # steps_count=REWARD_STEPS：unroll the Bellman equation for 10 steps
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE) # Adam优化

    total_rewards = [] # 前几局的奖励之和
    step_rewards = []
    step_idx = 0 # 迭代次数/轮数
    done_episodes = 0 # 局数，几局游戏
    reward_sum = 0.0 # the sum of the discounted reward for every transition
    bs_smoothed = entropy = l_entropy = l_policy = l_total = None

    batch_states, batch_actions, batch_scales = [], [], []

    for step_idx, exp in enumerate(exp_source): # (state, action, reward, last_state)
        reward_sum += exp.reward # the sum of the discounted reward for every transition
        baseline = reward_sum / (step_idx + 1) # 奖励除以迭代次数(步数) the baseline for the policy scale
        writer.add_scalar("baseline", baseline, step_idx)
        batch_states.append(exp.state) # 状态
        batch_actions.append(int(exp.action)) # 动作
        batch_scales.append(exp.reward - baseline) # 优势函数，引入基准

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards() # 返回一局游戏过后的total_rewords
        if new_rewards:
            done_episodes += 1 # 游戏局数(回合数)
            reward = new_rewards[0] # 本局游戏奖励
            total_rewards.append(reward) # 前几局的奖励之和
            mean_rewards = float(np.mean(total_rewards[-100:])) # 前几局的奖励之和/回合数
            print("第%d次: 第%d局游戏结束, 奖励为%6.2f, 平均奖励为%6.2f" % (step_idx, done_episodes, reward, mean_rewards))
            writer.add_scalar("reward", reward, step_idx) # 本局游戏奖励
            writer.add_scalar("reward_100", mean_rewards, step_idx) # 前几局游戏的平均奖励
            writer.add_scalar("episodes", done_episodes, step_idx) # 游戏局数(回合数)
            if mean_rewards > 50: # 最大期望奖励阈值，只有当mean_rewards > 50时才结束游戏
                print("经过%d轮完成%d局游戏!" % (step_idx, done_episodes))
                break

        if len(batch_states) < BATCH_SIZE: # state里面还没超过BATCH_SIZE个样本
            continue

        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales) # 优势函数，引入基准

        # train
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

        # 下面实际上没有平滑，返回实际值(这5行代码在本程序中没意义)
        bs_smoothed = smooth(bs_smoothed, np.mean(batch_scales))
        entropy = smooth(entropy, entropy_v.item())
        l_entropy = smooth(l_entropy, entropy_loss_v.item())
        l_policy = smooth(l_policy, loss_policy_v.item())
        l_total = smooth(l_total, loss_v.item())

        writer.add_scalar("baseline", baseline, step_idx)
        writer.add_scalar("entropy", entropy, step_idx)
        writer.add_scalar("loss_entropy", l_entropy, step_idx)
        writer.add_scalar("loss_policy", l_policy, step_idx)
        writer.add_scalar("loss_total", l_total, step_idx)
        writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
        writer.add_scalar("grad_max", grad_max, step_idx)
        writer.add_scalar("batch_scales", bs_smoothed, step_idx)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()
