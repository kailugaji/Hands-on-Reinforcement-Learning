#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# Q-learning for FrozenLake
# https://www.cnblogs.com/kailugaji/
# 与上一个值迭代法相比，这个版本使用了更多的迭代来解决问题。
# 其原因是不再使用测试过程中获得的经验。
# 在上一个Q迭代例子中，周期性的测试会引起Q表统计的更新。
# 本算法在测试过程中不接触Q值，这在环境得到解决之前会造成更多的迭代。
# 总的来说，环境所需的样本总数几乎是一样的。
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9 # 折扣率
ALPHA = 0.2 # 平滑指数
TEST_EPISODES = 20 # 玩几局游戏

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self): # 随机采样动作
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state): # 从Q表中选择最优值与动作
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s): # 平滑
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v # r(s, a, s') + γ * max Q(s, a)
        old_v = self.values[(s, a)]
        self.values[(s, a)] = old_v * (1-ALPHA) + new_v * ALPHA # 这变了，Q值平滑收敛
        # Q(s, a) <- (1-α) * Q(s, a) + α * (r(s, a, s') + γ * max Q(s, a))

    def play_episode(self, env): # 玩一局游戏
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state) # 给定状态，从Q表中选择最优动作
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env() # 执行一个随机步骤
        agent.value_update(s, a, r, next_s)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()