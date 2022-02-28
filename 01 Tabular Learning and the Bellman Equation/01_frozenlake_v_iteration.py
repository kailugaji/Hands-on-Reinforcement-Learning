#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# Value iteration for FrozenLake
# https://www.cnblogs.com/kailugaji/
import gym
import collections
from tensorboardX import SummaryWriter
import time

ENV_NAME = "FrozenLake-v1" #游戏环境
'''
S: initial stat 起点
F: frozen lake 冰湖
H: hole 窟窿
G: the goal 目的地
agent要学会从起点走到目的地，并且不要掉进窟窿
'''
GAMMA = 0.9 # 折扣率
TEST_EPISODES = 20 # 玩几局游戏

class Agent: #保存表格，并包含将在训练循环中使用的函数
    def __init__(self):
        self.env = gym.make(ENV_NAME) #创建游戏环境
        self.state = self.env.reset() # 用于重置环境
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    '''
    此功能用于从环境中收集随机经验，并更新奖励和过渡表。
    注意，我们不需要等到一局游戏(回合)结束才开始学习；
    我们只需执行N个步骤，并记住它们的结果。
    这是值迭代和交叉熵方法的区别之一，交叉熵方法只能在完整的回合中学习。
    '''
    def play_n_random_steps(self, count): # 玩100步，得到回报表与转换表
        for _ in range(count):
            action = self.env.action_space.sample()  # 随机采样选择动作
            new_state, reward, is_done, _ = self.env.step(action) # 根据动作，与环境互动得到的新的状态与奖励
            self.rewards[(self.state, action, new_state)] = reward # 回报表：源状态，动作，目标状态
            self.transits[(self.state, action)][new_state] += 1 # 转换表：状态，动作，新状态的概率
            self.state = self.env.reset() if is_done else new_state

    def calc_action_value(self, state, action): # 步骤5：给定s, a, 计算Q(s, a)
        target_counts = self.transits[(state, action)] # 转换表：状态，动作
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)] # 回报表：源状态，动作，目标状态
            val = reward + GAMMA * self.values[tgt_state] # 值表只有一个：目标状态
            action_value += (count / total) * val # 期望值——状态动作值函数(Q值)
        return action_value # Q值

    def select_action(self, state): # 步骤6：给定状态，找最优动作
        best_action, best_value = None, None
        for action in range(self.env.action_space.n): # 遍历所有动作
            action_value = self.calc_action_value(state, action) # 步骤5：Q值
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action # 找使Q值最大的那个动作——最优动作 a = argmax Q(s, a)

    def play_episode(self, env): # 玩一局游戏
        total_reward = 0.0
        state = env.reset() # 用于重置环境
        while True:
            action = self.select_action(state) # 步骤6：最优动作
            # 不同于"Windows下OpenAI gym环境的使用"中的随机采样动作
            new_state, reward, is_done, _ = env.step(action) # 根据动作，与环境交互得到的新的状态与奖励
            self.rewards[(state, action, new_state)] = reward # 更新表
            self.transits[(state, action)][new_state] += 1 # 转换表
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward # 得到一局游戏过后的总体奖励

    def value_iteration(self): # 值迭代循环
        # 用s状态下可用的动作的最大值来更新当前状态的值
        # 任意s，π(s) = arg max Q(s, a)
        for state in range(self.env.observation_space.n): # 步骤2-4：遍历状态空间，找使Q值最大的最优策略
            state_values = [
                self.calc_action_value(state, action) # 计算Q(s, a)
                for action in range(self.env.action_space.n) # 遍历动作空间
            ]
            self.values[state] = max(state_values) # 步骤3：对于每个状态，V(s) = max Q(s, a)
            # 更新V值表，最优状态值函数，贝尔曼最优方程

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True: # 重复试验，直到20局游戏的平均奖励大于0.8，迭代终止
        iter_no += 1 # iter_no：重复试验的迭代次数
        agent.play_n_random_steps(100) # 步骤1：每一局游戏执行100个随机步骤，填充回报和转换表
        agent.value_iteration() # 步骤2-4：100步之后，对所有的状态进行一次值迭代循环，更新V值表，作为策略
        # time.sleep(0.1) #为了让显示变慢，否则画面会非常快
        # test_env.render() # 用于渲染出当前的智能体以及环境的状态

        reward = 0.0
        for _ in range(TEST_EPISODES): # 玩20局游戏
            reward += agent.play_episode(test_env) # 用到步骤5-6, 20局游戏奖励之和
        reward /= TEST_EPISODES # 20局的平均奖励
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward # 找到最优的奖励
        if reward > 0.80: # 重复试验次数，直到奖励>0.8，停止迭代
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
