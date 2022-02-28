#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# The PTAN library——Experience replay buffers 经验回放池
# 在DQN中，很少处理即时的经验样本，因为它们是高度相关的，这导致了训练中的不稳定性
# 构建一个很大的经验回放池，其中填充了经验片段
# 然后对回放池进行采样(随机或带优先级权重)，得到训练批。
# 经验回放池通常有最大容量，所以当经验回放池达到极限时，旧的样本将被推出。
# 训练时，随机从经验池中抽取样本来代替当前的样本用来进行训练。
# 这样,就打破了和相邻训练样本的相似性,避免模型陷入局部最优
# https://www.cnblogs.com/kailugaji/
import gym
import ptan
from typing import List, Optional, Tuple, Any

# 构建Environment
class ToyEnv(gym.Env):
    """
    Environment with observation 0..4 and actions 0..2
    Observations are rotated sequentialy mod 5, reward is equal to given action.
    Episodes are having fixed length of 10
    """
    def __init__(self):
        super(ToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=5) # integer observation, which increases from 0 to 4
        self.action_space = gym.spaces.Discrete(n=3) # integer action, which increases from 0 to 2
        self.step_index = 0

    def reset(self):
        self.step_index = 0
        return self.step_index

    def step(self, action):
    # 输入：action
    # 输出：observation, reward, done, info
        is_done = self.step_index == 10 # 一局游戏走10步
        if is_done:
            return self.step_index % self.observation_space.n, 0.0, is_done, {}
        self.step_index += 1
        reward = float(action)
        return self.step_index % self.observation_space.n, reward, self.step_index == 10, {}
        # Observation: mod 5, 0-4一循环，依次递增

# 构建Agent
class DullAgent(ptan.agent.BaseAgent):
    """
    Agent always returns the fixed action
    """
    def __init__(self, action: int):
        self.action = action

    def __call__(self, observations: List[Any], state: Optional[List] = None) -> Tuple[List[int], Optional[List]]:
        # 不管observations输入的是什么，结果都是输入的action的值
        return [self.action for _ in observations], state


if __name__ == "__main__":
    env = ToyEnv()
    agent = DullAgent(action=1) # 生成固定动作，与action的取值保持一致，与observations取值无关
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=1)
    # 输出的信息格式为：(state, action, reward, last_state)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=100)
    # a simple replay buffer of predefined size with uniform sampling.
    # 构建buffer，容量为100，当前没东西，len(buffer) = 0

    for step in range(6): # 最大buffer进6个样本
        buffer.populate(1) # 从环境中获取一个新样本
        # The method populate(N) to get N samples from the experience source and put them into the buffer
        print("第%d次buffer大小：" %step, len(buffer))
        if len(buffer) < 5: # buffer里面还没超过5个样本
            continue # if buffer is small enough (<5), do nothing
        # buffer等于或超过5个后，从buffer里面均匀抽样一个批次的样本，一批4个样本
        batch = buffer.sample(4) # The method sample(N) to get the batch of N experience objects
        print("Train time, %d batch samples:" % len(batch))
        for s in batch:
            print(s)
