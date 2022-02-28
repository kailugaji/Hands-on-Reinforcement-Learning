#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# The PTAN library——The PTAN CartPole solver
# 前述5个程序全部是为了CartPole实战做准备
# https://www.cnblogs.com/kailugaji/
import gym
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pylab as plt
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun']
}
rcParams.update(config)


HIDDEN_SIZE = 128 # 隐层神经元个数
BATCH_SIZE = 16 # 一批16个样本
TGT_NET_SYNC = 10 #每隔10轮将参数从原网络同步到目标网络
GAMMA = 0.9 # 折扣率
REPLAY_SIZE = 1000 # 经验回放池容量
LR = 5e-3 # 学习率
EPS_DECAY=0.995 # epsilon因子线性衰减率

# 构建网络
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        # obs_size：输入状态维度，hidden_size：隐层维度，n_actions：输出动作维度
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size), # 全连接层
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions) # 全连接层
        )

    def forward(self, x):
    # CartPole is stupid -- they return double observations, rather than standard floats, so, the cast here
        return self.net(x.float())


@torch.no_grad() # 下面数据不需要计算梯度，也不会进行反向传播
def unpack_batch(batch, net, gamma):
# batch: 一批次的样本，16个，(state, action, reward, last_state)
    states = []
    actions = []
    rewards = []
    done_masks = []
    last_states = []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        done_masks.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)

    states_v = torch.tensor(states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    last_states_v = torch.tensor(last_states)
    last_state_q_v = net(last_states_v) # 将最后的状态输入网络，得到Q(s, a)
    best_last_q_v = torch.max(last_state_q_v, dim=1)[0] # 找最大的Q
    best_last_q_v[done_masks] = 0.0
    return states_v, actions_v, best_last_q_v * gamma + rewards_v
    # r + gamma * max Q(s, a)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    obs_size = env.observation_space.shape[0]
    # observation大小(4个状态变量)：小车在轨道上的位置，杆子与竖直方向的夹角，小车速度，角度变化率
    n_actions = env.action_space.n # action大小(2个动作，左或者右)

    net = Net(obs_size, HIDDEN_SIZE, n_actions) # 4->128->2
    tgt_net = ptan.agent.TargetNet(net) # 目标网络(与原网络架构一致)
    selector = ptan.actions.ArgmaxActionSelector() # 选Q值最大的动作索引
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1, selector=selector)
    # epsilon-greedy action selector，初始epsilon=1
    agent = ptan.agent.DQNAgent(net, selector) # 离散：输出具有最大Q值的动作与状态索引
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    # 返回运行记录以用于训练模型，输出格式为：(state, action, reward, last_state)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    # 经验回放池，构建buffer，容量为1000，当前没东西，len(buffer) = 0
    optimizer = optim.Adam(net.parameters(), LR) # Adam优化

    step = 0 # 迭代次数/轮数
    episode = 0 # 局数，几局游戏
    solved = False
    losses = []
    rewards = []

    while True:
        step += 1
        buffer.populate(1) # 从环境中获取一个新样本

        for reward, steps in exp_source.pop_rewards_steps():
        # pop_rewards_steps(): 返回一局游戏过后的（total_reword，total_steps）
            episode += 1
            print("第%d次: 第%d局游戏结束, 奖励为%.2f, 本局步数为%d, epsilon为%.2f" %(step, episode, reward, steps, selector.epsilon))
            # 杆子能越长时间保持平衡,得分越高。steps与reward一致
            rewards.append(reward)
            solved = reward > 100 # 最大奖励阈值，只有当reward>100时才结束游戏
        if solved:
            print("Victory!")
            break

        # print("第%d次buffer大小：" % step, len(buffer))
        if len(buffer) < 2*BATCH_SIZE: # # buffer里面还没超过2倍的批大小(32)个样本
            continue

        batch = buffer.sample(BATCH_SIZE)
        # buffer等于或超过2*BATCH_SIZE后，从buffer里面均匀抽样一个批次的样本，一批BATCH_SIZE个样本
        # batch：state, action, reward, last_state
        states_v, actions_v, tgt_q_v = unpack_batch(batch, tgt_net.target_model, GAMMA)
        # 输入目标网络
        # 得到tgt_q_v = r + gamma * max Q(s, a)
        optimizer.zero_grad()
        q_v = net(states_v) # 输入状态，得到Q(s, a)
        q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        '''
            torch.gather 作用：收集输入的特定维度指定位置的数值
            参数：input(tensor):   待操作数。不妨设其维度为（x1, x2, …, xn）
                  dim(int):   待操作的维度。
                  index(LongTensor):   如何对input进行操作。
                  其维度有限定，例如当dim=i时，index的维度为（x1, x2, …y, …,xn），既是将input的第i维的大小更改为y，且要满足y>=1（除了第i维之外的其他维度，大小要和input保持一致）。
                  out:   注意输出和index的维度是一致的
            squeeze(-1): 将输入张量形状中的1去除并返回。
            如果输入是形如(A×1×B×1×C×1×D)，那么输出形状就为： (A×B×C×D)
        '''
        loss_v = F.mse_loss(q_v, tgt_q_v)
        # MSE Loss, min L = (r + gamma * max Q(s', a') - Q(s, a))^2
        loss_v.backward()
        optimizer.step()
        losses.append(loss_v.item())
        selector.epsilon *= EPS_DECAY # 贪心因子线性衰减

        if step % TGT_NET_SYNC == 0: # 每TGT_NET_SYNC(10)轮同步一次目标网络参数
            tgt_net.sync() # weights from the source network are copied into the target network

    # 画图
    # Loss曲线图
    plt.plot(losses)
    plt.xlabel('Iteration', fontsize=13) # 迭代次数
    plt.ylabel('Loss', fontsize=13)
    plt.title('CartPole-v0', fontsize=14)
    plt.savefig('损失函数曲线图.png', dpi=1000)
    plt.show()
    # reward曲线图
    plt.plot(rewards)
    plt.xlabel('Episode', fontsize=13) # 几局游戏
    plt.ylabel('Reward', fontsize=13)
    plt.title('CartPole-v0', fontsize=14)
    plt.savefig('奖励曲线图.png', dpi=1000)
    plt.show()