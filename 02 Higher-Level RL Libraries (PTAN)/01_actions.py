#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# The PTAN library——Action selectors 动作选择器
# 从网络输出(Q值)到具体的动作值
# https://www.cnblogs.com/kailugaji/
import ptan
import numpy as np

if __name__ == "__main__":
    print("方法1：基于值函数的方法 (网络输出的是Q值)")
    q_vals_1 = np.array([
        [1, 2, 3],
        [1, -1, 0]
    ]) # 事先定义网络输出的Q值
    print("Q值：\n", q_vals_1)

    selector = ptan.actions.ArgmaxActionSelector()
    print("具有最大Q值的动作索引：", selector(q_vals_1))
    # 返回具有最大Q值的动作的索引——[列，行]

    print("采用epsilon贪心策略的动作索引：")
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.0) # 以epsilon的概率随机选择值
    print("当epsilon=0.0：", selector(q_vals_1)) # no random actions

    selector.epsilon = 1.0 # will be random
    print("当epsilon=1.0：", selector(q_vals_1))

    selector.epsilon = 0.5
    print("当epsilon=0.5：", selector(q_vals_1))

    selector.epsilon = 0.1
    print("当epsilon=0.1：", selector(q_vals_1))

    print("-----------------------------------------------------")
    print("方法2：基于策略函数的方法 (网络输出的是标准化概率分布)")
    print("从三个概率分布中采样得到的动作：")
    q_vals_2 = np.array([
        [0.1, 0.8, 0.1], # 分布0 # 行归一化
        [0.0, 0.0, 1.0], # 分布1
        [0.5, 0.5, 0.0]  # 分布2
    ]) # 事先定义网络输出的概率分布
    # 从三个分布中进行抽样:
    # 在第一个分布中，选择索引为1的动作的概率为80%
    # 在第二个分布中，总是选择索引为2的动作
    # 在第三个分布中，选择索引为0的动作和索引为1的动作是等可能的
    selector = ptan.actions.ProbabilityActionSelector()
    # 从概率分布中采样 (输入必须是一个标准化的概率分布)
    for i in range(8): # 采样8次
        acts = selector(q_vals_2)
        print('第 %d 次: ' %(i+1), acts)
        # acts的三个值分别是从三个分布中采样的动作的索引
        # 可以看到第二个值始终是2，这是因为第二个分布中索引为2的动作的概率为1
