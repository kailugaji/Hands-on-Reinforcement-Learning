#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# Policy Gradients——REINFORCE
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

# Calculate the reward from the end of the local reward list.
# Indeed, the last step of the episode will have a total reward equal to its local reward.
# The step before the last will have the total reward of r(𝑡−1) + gamma * r(𝑡)
# sum(gamma^t * reward)
def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards): # reversed: 返回反向的迭代器对象
        sum_r *= GAMMA # gamma * r(𝑡), r(t): the total reward for the previous steps
        sum_r += r # r(𝑡−1) + gamma * r(𝑡), r(t-1): the local reward
        res.append(sum_r) # the discounted total reward
    return list(reversed(res)) # a list of rewards for the whole episode


if __name__ == "__main__":
    env = gym.make("CartPole-v0") # 创建游戏环境
    writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PGN(env.observation_space.shape[0], env.action_space.n) # 4(状态)->128->2(动作)
    # print(net)

    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)
    # PolicyAgent：连续
    # make a decision about actions for every observation (依概率)
    # apply_softmax=True：网络输出先经过softmax转化成概率，再从这个概率分布中进行随机抽样
    # float32_preprocessor：returns the observation as float64 instead of the float32 required by PyTorch
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    # 返回运行记录以用于训练模型，输出格式为：(state, action, reward, last_state)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE) # Adam优化

    total_rewards = [] # the total rewards for the episodes
    step_idx = 0 # 迭代次数/轮数
    done_episodes = 0 # 局数，几局游戏

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = [] # local rewards for the episode being currently played

    for step_idx, exp in enumerate(exp_source): # (state, action, reward, last_state)
        batch_states.append(exp.state) # 状态
        batch_actions.append(int(exp.action)) # 动作
        cur_rewards.append(exp.reward) # 即时奖励

        if exp.last_state is None: # 一局游戏结束
            batch_qvals.extend(calc_qvals(cur_rewards)) # the discounted total rewards
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

        # train
        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals) # the discounted batch total rewards

        logits_v = net(states_v) # 输入状态，输出Q(s, a)值
        log_prob_v = F.log_softmax(logits_v, dim=1) # 输出动作的对数概率分布 log π(s, a)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        # max sum(gamma^t * reward) * log π(s, a)
        loss_v = -log_prob_actions_v.mean() # min

        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()
