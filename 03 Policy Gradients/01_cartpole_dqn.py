#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# DQN
# The CartPole example
# https://www.cnblogs.com/kailugaji/
import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

GAMMA = 0.99 # 折扣率 0.9
LEARNING_RATE = 0.01 # 学习率 5e-3
BATCH_SIZE = 16 # 一批xx个样本 16

EPSILON_START = 1.0 # epsilon因子
EPSILON_STOP = 0.02
EPSILON_STEPS = 5000

REPLAY_BUFFER = 5000 # 经验回放池容量

# 构建网络
class DQN(nn.Module):
    def __init__(self, input_size, n_actions):
        # input_size：输入状态维度，hidden_size：隐层维度=128，n_actions：输出动作维度
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128), # 全连接层，隐层预设为128维度
            nn.ReLU(),
            nn.Linear(128, n_actions) # 全连接层
        )

    def forward(self, x):
        return self.net(x)

# 目标网络 r + gamma * max Q(s, a)
def calc_target(net, local_reward, next_state):
    if next_state is None:
        return local_reward
    state_v = torch.tensor([next_state], dtype=torch.float32)
    next_q_v = net(state_v) # 将最后的状态输入网络，得到Q(s, a)
    best_q = next_q_v.max(dim=1)[0].item() # 找最大的Q
    return local_reward + GAMMA * best_q
    # r + gamma * max Q(s, a)


if __name__ == "__main__":
    env = gym.make("CartPole-v0") # 创建游戏环境
    writer = SummaryWriter(comment="-cartpole-dqn")

    net = DQN(env.observation_space.shape[0], env.action_space.n) # 4(状态)->128->2(动作)
    # print(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
    # epsilon-greedy action selector，初始epsilon=1
    agent = ptan.agent.DQNAgent(net, selector, preprocessor=ptan.agent.float32_preprocessor)
    # DQNAgent：离散
    # dqn_model换成自定义的DQN模型，4(状态)->128->2(动作)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    # 返回运行记录以用于训练模型，输出格式为：(state, action, reward, last_state)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_BUFFER)
    # 经验回放池，构建buffer，容量为1000，当前没东西，len(buffer) = 0
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE) # Adam优化
    mse_loss = nn.MSELoss() # MSE loss

    total_rewards = [] # the total rewards for the episodes
    step_idx = 0 # 迭代次数/轮数
    done_episodes = 0 # 局数，几局游戏

    while True:
        step_idx += 1
        selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)
        # epsilon随迭代步数的衰减策略
        replay_buffer.populate(1)  # 从环境中获取一个新样本

        if len(replay_buffer) < BATCH_SIZE: # buffer里面还没超过BATCH_SIZE个样本
            continue

        # sample batch
        batch = replay_buffer.sample(BATCH_SIZE) # 从buffer里面均匀抽样一个批次的样本，一批BATCH_SIZE个样本
        batch_states = [exp.state for exp in batch]
        batch_actions = [exp.action for exp in batch]
        batch_targets = [calc_target(net, exp.reward, exp.last_state)
                         for exp in batch] # r + gamma * max Q(s, a)
        # train
        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        net_q_v = net(states_v) # 输入状态，输出Q(s, a)值
        target_q = net_q_v.data.numpy().copy() # copy网络参数
        target_q[range(BATCH_SIZE), batch_actions] = batch_targets
        target_q_v = torch.tensor(target_q) # r + gamma * max Q(s, a)
        loss_v = mse_loss(net_q_v, target_q_v) # min L = (r + gamma * max Q(s', a') - Q(s, a))^2
        loss_v.backward()
        optimizer.step()

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards() # 返回一局游戏过后的total_rewords
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward) # 前done_episodes局游戏的奖励之和
            mean_rewards = float(np.mean(total_rewards[-100:])) # total_rewards/done_episodes
            print("第%d次: 第%d局游戏结束, 奖励为%6.2f, 平均奖励为%6.2f, epsilon为%.2f" % (step_idx, done_episodes, reward, mean_rewards, selector.epsilon))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("epsilon", selector.epsilon, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 50: # 最大期望奖励阈值，只有当平均奖励 > 50时才结束游戏
                print("经过%d轮完成%d局游戏!" % (step_idx, done_episodes))
                break
    writer.close()
