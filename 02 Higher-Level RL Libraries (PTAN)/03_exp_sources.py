#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# The PTAN library——Experience source
# 是对智能体在环境中运行过程的一种封装，屏蔽了很多运行细节，最终只返回运行记录以用于训练模型
# 常用的两个封装类有：ExperienceSource，ExperienceSourceFirstLast(推荐使用)
# 部分参考： https://blog.csdn.net/HJJ19881016/article/details/105743835/
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

    def reset(self): # 用于重置环境
        self.step_index = 0
        return self.step_index

    def step(self, action):
    # 输入：action
    # 输出：observation, reward, done, info
    # observation（object）一个特定的环境对象，代表了你从环境中得到的观测值
    # reward（float）由于之前采取的动作所获得的大量奖励，与环境交互的过程中，奖励值的规模会发生变化，但是总体的目标一直都是使得总奖励最大
    # done（boolean）决定是否将环境初始化，大多数，但不是所有的任务都被定义好了什么情况该结束这个回合
    # info（dict）调试过程中将会产生的有用信息，有时它会对我们的强化学习学习过程很有用
        is_done = self.step_index == 10 # 一局游戏走10步
        if is_done:
            return self.step_index % self.observation_space.n, 0.0, is_done, {}
        # Observation: mod 5, 0-4一循环，依次递增
        self.step_index += 1
        reward = float(action)
        return self.step_index % self.observation_space.n, reward, self.step_index == 10, {}
        # 这里定义了reward = action，info = {}, 玩够10步done=True

# 构建Agent
# 继承BaseAgent来自定义自己的Agent类，通过重写__call__()方法来实现Obervation到action的转换逻辑
class DullAgent(ptan.agent.BaseAgent):
    """
    Agent always returns the fixed action
    """
    def __init__(self, action: int):
        self.action = action

    def __call__(self, observations: List[Any], state: Optional[List] = None) -> Tuple[List[int], Optional[List]]:
    # "->"常常出现在python函数定义的函数名后面，为函数添加元数据，描述函数的返回类型，从而方便开发人员使用
    # 不管observations输入的是什么，结果都是输入的action的值
        return [self.action for _ in observations], state


if __name__ == "__main__":
    print("案例I：")
    env = ToyEnv()
    s = env.reset()
    print("env.reset() -> %s" % s)
    s = env.step(1) # action = 1
    print("env.step(1) -> %s" % str(s))
    s = env.step(2) # action = 2
    print("env.step(2) -> %s" % str(s))
    # 输出：observation, reward, done, info

    for i in range(10):
        r = env.step(0) # action = 0
        print("第 %d 次 env.step(0) -> %s" % (i, str(r)))
    # 重复10次，action的索引为0
    # 输出：observation, reward, done, info

    print("-------------------------------------------------------------------")
    print("案例II：")
    agent = DullAgent(action=1) # 生成固定动作，与action的取值保持一致，与observations取值无关
    print("agent:", agent(observations=[2, 1, 3, 1])[0])
    # [1, 2]: observations
    # [0]只输出动作索引

    print("-------------------------------------------------------------------")
    print("案例III：")
    env = ToyEnv()
    agent = DullAgent(action=1) # 生成固定动作，始终为1
    print("1. ExperienceSource (steps_count=2): ")
    exp_source_1 = ptan.experience.ExperienceSource(env, agent, steps_count=2)
    # ExperienceSource输入：
    # env: The Gym environment to be used. Alternatively, it could be the list of environments.
    # agent: The agent instance.
    # steps_count: 用于说明一条记录中包含的步(step)数 (sub-trajectories of length 2)
    # ExperienceSource输出：
    # 返回智能体在环境中每一步的交互信息，输出格式为：(state, action, reward, done)
    # 其中state为agent所处的状态，action为采取的动作，reward为采取action后获得的即时奖励，done用来标识episode是否结束。
    for idx, exp in enumerate(exp_source_1):
        if idx > 15:
            break
        print("第%d步" %(idx), exp)

    print("2. ExperienceSource (steps_count=4): ")
    exp_source_2 = ptan.experience.ExperienceSource(env, agent, steps_count=4)
    # print(next(iter(exp_source_2))) # 只一步
    # iter()返回迭代器对象
    # next()函数自动调用文件第一行并返回下一行
    for idx, exp in enumerate(exp_source_2):
        if exp[0].done:
            break
        print("第%d步" %(idx), exp)

    print("3. ExperienceSource (steps_count=2): ")
    exp_source_3 = ptan.experience.ExperienceSource([ToyEnv(), ToyEnv()], agent, steps_count=2)
    # 环境正在以循环的方式迭代，从两个环境中一步步获取轨迹。
    for idx, exp in enumerate(exp_source_3):
        if idx > 20:
            break
        print("第%d步" %(idx), exp)

    print("4. ExperienceSourceFirstLast (steps_count=1): ")
    exp_source_4 = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=1)
    # 输出的信息格式为：(state, action, reward, last_state)
    # 并不会输出每一步的信息，而是把多步的交互结果综合(累计多步的reward;显示头尾的状态)到一条Experience输出
    # 多步rewards的累加是有衰退的，而其中的衰退系数由参数gamma(折扣率)指定，即reward=r1+gamma∗r2+(gamma^2)∗r3
    # 其中rn代表第n步操作获得的reward
    # last_state: the state we've got after executing the action. If our episode ends, we have None here
    for idx, exp in enumerate(exp_source_4):
        print("第%d步" %(idx), exp)
        if idx > 10:
            break

    print("5. ExperienceSourceFirstLast (steps_count=4): ")
    exp_source_5 = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=0.6, steps_count=4)
    # 输出的信息格式为：(state, action, reward, last_state)
    # 并不会输出每一步的信息，而是把多步的交互结果综合(累计多步的reward;显示头尾的状态)到一条Experience输出
    # 多步rewards的累加是有衰退的，而其中的衰退系数由参数gamma指定，即reward=r1+gamma∗r2+(gamma^2)∗r3
    # 其中rn代表第n步操作获得的reward
    # last_state: the state we've got after executing the action. If our episode ends, we have None here
    for idx, exp in enumerate(exp_source_5):
        print("第%d步" % (idx), exp)
        if idx > 10:
            break
