#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# Policy Gradients——带基准线的REINFORCE算法(REINFORCE with Baseline)
# REINFORCE算法缺点：不同路径之间的方差大，导致训练不稳定
# 引入一控制变量sum(gamma^t * reward)的期望：E(sum(gamma^t * reward))，以减小方差
# Baseline：the mean of the discounted rewards
# The CartPole example
# https://www.cnblogs.com/kailugaji/
import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99 # 折扣率
LEARNING_RATE = 0.01 # 学习率
EPISODES_TO_TRAIN = 4 # how many complete episodes we will use for training

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

# 与上一个程序唯一的不同在这！！！
# sum(gamma^t * reward) - E(sum(gamma^t * reward))
def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    res = list(reversed(res))
    mean_q = np.mean(res)
    return [q - mean_q for q in res]


if __name__ == "__main__":
    env = gym.make("CartPole-v0") # 创建游戏环境
    writer = SummaryWriter(comment="-cartpole-reinforce-baseline")

    net = PGN(env.observation_space.shape[0], env.action_space.n) # 4(状态)->128->2(动作)
    # print(net)

    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,apply_softmax=True)
    # PolicyAgent：连续
    # make a decision about actions for every observation (依概率)
    # apply_softmax=True：网络输出先经过softmax转化成概率，再从这个概率分布中进行随机抽样
    # float32_preprocessor：returns the observation as float64 instead of the float32 required by PyTorch
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    # 返回运行记录以用于训练模型，输出格式为：(state, action, reward, last_state)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE) # Adam优化

    total_rewards = [] # 前几局的奖励之和
    step_idx = 0 # 迭代次数/轮数
    done_episodes = 0 # 局数，几局游戏

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_states, cur_actions, cur_rewards = [], [], []

    for step_idx, exp in enumerate(exp_source): # (state, action, reward, last_state)
        cur_states.append(exp.state) # 状态
        cur_actions.append(int(exp.action)) # 动作
        cur_rewards.append(exp.reward) # 即时奖励

        if exp.last_state is None: # 一局游戏结束
            batch_states.extend(cur_states)
            batch_actions.extend(cur_actions)
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_states.clear()
            cur_actions.clear()
            cur_rewards.clear()
            batch_episodes += 1

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards() # 返回一局游戏过后的total_rewords
        if new_rewards:
            done_episodes += 1 # 游戏局数(回合数)
            reward = new_rewards[0]
            total_rewards.append(reward) # the total rewards for the episodes
            mean_rewards = float(np.mean(total_rewards[-100:])) # 平均奖励
            print("第%d次: 第%d局游戏结束, 奖励为%6.2f, 平均奖励为%6.2f" % (step_idx, done_episodes, reward, mean_rewards))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 50: # 最大期望奖励阈值，只有当mean_rewards > 50时才结束游戏
                print("经过%d轮完成%d局游戏!" % (step_idx, done_episodes))
                break

        if batch_episodes < EPISODES_TO_TRAIN: # how many complete episodes we will use for training
            continue

        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        # train
        optimizer.zero_grad()
        logits_v = net(states_v) # 输入状态，输出Q(s, a)值
        log_prob_v = F.log_softmax(logits_v, dim=1) # 输出动作的对数概率分布 log π(s, a)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        # max (sum(gamma^t * reward) - E(sum(gamma^t * reward))) * log π(s, a)
        loss_v = -log_prob_actions_v.mean() # min

        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()
