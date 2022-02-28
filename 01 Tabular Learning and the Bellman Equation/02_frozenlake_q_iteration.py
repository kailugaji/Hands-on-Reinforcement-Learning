#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# Q-learning for FrozenLake
# 1. 值表变了。上例保留了状态的值，因此字典中的键只是一个状态。
# 现在需要存储Q函数的值，它有两个参数：状态和动作，因此值表中的键现在是复合键。
# 2. 不需要calc_action_value()函数。因为我们的动作值存储在值表中。
# 3. value_iteration()变了。
# https://www.cnblogs.com/kailugaji/
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1" # 游戏环境
'''
S: initial stat 起点
F: frozen lake 冰湖
H: hole 窟窿
G: the goal 目的地
agent要学会从起点走到目的地，并且不要掉进窟窿
'''
GAMMA = 0.9 # 折扣率
TEST_EPISODES = 20 # 玩几局游戏


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME) # 创建游戏环境
        self.state = self.env.reset() # 用于重置环境
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count): # 玩100步，得到回报表与转换表
        for _ in range(count):
            action = self.env.action_space.sample() # 随机采样选择动作
            new_state, reward, is_done, _ = self.env.step(action) # 根据动作，与环境互动得到的新的状态与奖励
            self.rewards[(self.state, action, new_state)] = reward # 回报表：源状态，动作，目标状态
            self.transits[(self.state, action)][new_state] += 1 # 转换表：状态，动作
            self.state = self.env.reset() if is_done else new_state

    def select_action(self, state): # 给定状态s, a = argmax Q(s, a)
        best_action, best_value = None, None
        for action in range(self.env.action_space.n): # 遍历所有动作
            action_value = self.values[(state, action)] # Q值表里有两个：状态与动作
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action # 直接建立Q表，从Q值表里找最优动作

    def play_episode(self, env): # 玩一局游戏
        total_reward = 0.0
        state = env.reset() # 用于重置环境
        while True:
            action = self.select_action(state) # 给定状态s, 最优动作a = argmax Q(s, a)
            new_state, reward, is_done, _ = env.step(action) # 根据动作，与环境交互得到的新的状态与奖励
            self.rewards[(state, action, new_state)] = reward # 更新表
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state # 步骤8
        return total_reward # 得到一局游戏过后的总体奖励

    def value_iteration(self): # 变了
    # 选择具有最大Q值的动作，然后把这个Q值作为目标状态的值
        for state in range(self.env.observation_space.n):  # 步骤2-10：其中3：遍历状态空间
            for action in range(self.env.action_space.n): # 步骤4-9：遍历动作空间
                action_value = 0.0
                target_counts = self.transits[(state, action)] # 转换表：状态，动作
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    reward = self.rewards[(state, action, tgt_state)] # 回报表：源状态，动作，目标状态
                    best_action = self.select_action(tgt_state) # 给定状态s, 最优动作a = argmax Q(s, a)
                    val = reward + GAMMA * self.values[(tgt_state, best_action)] # 值表：目标状态，最优动作
                    action_value += (count / total) * val # 期望值——最优状态动作值函数(Q值)(其中动作为最优动作)
                    # 贝尔曼最优方程
                self.values[(state, action)] = action_value # 更新Q值表：状态，动作

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-iteration")

    iter_no = 0
    best_reward = 0.0
    while True: # 重复试验，直到20局游戏的平均奖励大于0.8，迭代终止
        iter_no += 1 # iter_no：重复试验的迭代次数
        agent.play_n_random_steps(100) # 步骤1：每一局游戏执行100个随机步骤，填充回报和转换表
        agent.value_iteration() # 步骤2-10：100步之后，对所有的状态进行一次值迭代循环，更新Q值表，作为策略
        # time.sleep(0.1) #为了让显示变慢，否则画面会非常快
        # test_env.render() # 用于渲染出当前的智能体以及环境的状态

        reward = 0.0
        for _ in range(TEST_EPISODES): # 玩20局游戏
            reward += agent.play_episode(test_env) # 20局游戏奖励之和
        reward /= TEST_EPISODES # 20局的平均奖励
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward # 找到最优的奖励
        if reward > 0.80: # 重复试验次数，直到奖励>0.8，停止迭代
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
