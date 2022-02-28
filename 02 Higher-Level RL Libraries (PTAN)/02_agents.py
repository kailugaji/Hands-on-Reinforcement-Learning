#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# The PTAN library——The agent
# https://www.cnblogs.com/kailugaji/
import ptan
import torch
import torch.nn as nn

# 方法1：基于值函数的方法 (网络输出的是Q值)
# DQNAgent
class DQNNet(nn.Module):
    def __init__(self, actions: int):
        super(DQNNet, self).__init__()
        self.actions = actions # 为简单起见，网络输出和输入一致, f(x)=x

    def forward(self, x):
        return torch.eye(x.size()[0], self.actions)
    # 定义了返回对角线全1，其余部分全0的二维数组，大小为(batch_size=x.size()[0], actions)

# 方法2：基于策略函数的方法 (网络输出的是标准化概率分布)
# PolicyAgent
class PolicyNet(nn.Module):
    def __init__(self, actions: int):
        super(PolicyNet, self).__init__()
        self.actions = actions # 为简单起见，网络输出和输入一致, f(x)=x

    def forward(self, x):
        # Now we produce the tensor with first two actions having the same logit scores
        shape = (x.size()[0], self.actions) # 大小为(batch_size=x.size()[0], actions)
        res = torch.zeros(shape, dtype=torch.float32)
        res[:, 0] = 1
        res[:, 1] = 1 # 定义了返回前两列为1，后面为0的二维数组
        return res


if __name__ == "__main__":
    net_1 = DQNNet(actions=3) # 3个动作(3列/3维)

    print("方法1：基于值函数的方法 (网络输出的是Q值)")
    net_in = torch.zeros(2, 10) # 输入2*10的全0矩阵，样本个数2，维度10
    net_out = net_1(net_in)
    print("DQN Net 输入：\n", net_in)
    print("DQN Net 输出：\n", net_out)
    # 得到对角线全1，其余部分全0的矩阵，大小为(batch_size=2, actions=3)

    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(dqn_model=net_1, action_selector=selector)
    # dqn_model换成自定义的DQNNet模型，action_selector保持不变，例子可见上一个程序01_actions.py
    ag_in = torch.zeros(2, 5) # 输入：2*5的全0矩阵，样本个数2，维度5 (a batch of two observations, each having five values)
    ag_out = agent(ag_in)
    print("DQN网络输入：\n", ag_in)
    print("具有最大Q值的动作与状态索引：", ag_out)
    # 输出动作与状态的索引
    # 1. 动作矩阵：网络输出中对应于1的动作索引，有2个样本，因此结果矩阵大小为1*2
    # 2. 状态列表：由于例子未涉及状态，因此为None

    print("采用epsilon贪心策略得到的动作索引：")
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.0) # no random actions
    agent = ptan.agent.DQNAgent(dqn_model=net_1, action_selector=selector)
    ag_in = torch.zeros(10, 5) # 输入：10*5的全0矩阵,10个样本
    ag_out = agent(ag_in)[0] # [0]表示只返回动作的索引，不返回状态的索引
    print("当epsilon=0:", ag_out) # DQNNet中actions=3使得第4维及后面索引全为0

    selector.epsilon = 1.0 # 当epsilon为1时，所有的动作都是随机的，与网络的输出无关
    ag_out = agent(ag_in)[0]
    print("当epsilon=1:", ag_out)

    selector.epsilon = 0.5
    ag_out = agent(ag_in)[0]
    print("当epsilon=0.5:", ag_out)

    selector.epsilon = 0.1
    ag_out = agent(ag_in)[0]
    print("当epsilon=0.1:", ag_out)

    print("----------------------------------------------------------------")
    net_2 = PolicyNet(actions=5) # 5个动作(5列)，0-4

    print("方法2：基于策略函数的方法 (网络输出的是标准化概率分布)")
    net_in = torch.zeros(6, 10) # 输入：6*10的全0矩阵,6个样本
    net_out = net_2(net_in)
    print("Policy Net 输入：\n", net_in)
    print("Policy Net 输出：\n", net_out)

    selector = ptan.actions.ProbabilityActionSelector()
    agent = ptan.agent.PolicyAgent(model=net_2, action_selector=selector, apply_softmax=True)
    # 对输出再采用softmax将数值归一化为[0, 1]的概率分布值
    ag_in = torch.zeros(6, 5) # 输入：6*5的全0矩阵,6个样本
    ag_out = agent(ag_in)[0]
    print("Policy网络输入：\n", ag_in)
    print("采样Policy方法得到的动作索引：", ag_out)
    # 采样索引为2-4的动作概率小于0与1
