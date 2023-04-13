#  起始探索：保证每一个状态-动作对都会作为初始状态-动作对出现在一些MDP样本序列中。

'''
 基于值迭代的起始探索每次访问蒙特卡罗强化学习算法
'''
## 价值迭代是一种利用迭代求解最优价值函数进而求解最优策略的方法。
## 策略评估迭代中，迭代算法利用Bellman期望方程迭代求解给定策略的价值函数；
## 价值迭代中，利用Bellman最优方程迭代求解最优策略的价值函数，进而求得最优策略。

## 起始探索的实现方法
# 1. 为每一个状态-动作对设置相同的初始出现概率
# 2. 循环所有状态-动作对   √
# 3. 规定状态-动作对初始出现次数

from collections import defaultdict
import numpy as np
import plot_func as plt
import blackjackEnv

class StartExplore_EveryVisit_ValueIter_MCRL():
    ## 类初始化
    def __init__(self, env, num_episodes=10000):
        self.env = env
        self.nA = env.action_space.n                          # 动作空间数量
        self.r_Q = defaultdict(lambda: np.zeros(self.nA))     # 动作价值函数
        # lambda是匿名函数，该字典返回的默认值就是一个长度为action_space.n的全0数组，也就是各状态初始时各动作的Q值都为0
        self.r_sum = defaultdict(lambda: np.zeros(self.nA))   # 累积折扣奖励之和
        self.r_count = defaultdict(lambda: np.zeros(self.nA))  # 累积折扣奖励次数
        self.policy = defaultdict(int)                         # 各状态下的策略
        self.num_episodes = num_episodes                       # 最大抽样次数

    ## 策略改进
    # 策略初始化及改进函数，初始化为：点数小于18则继续叫牌，否则停牌
    def update_policy(self, state):
        if state not in self.policy.keys():
            player, dealer, ace = state
            action = 0 if player >= 18 else 1   # 0: 停牌， 1：要牌
        else:
            action = np.argmax(self.r_Q[state])  # 最优动作价值对应的动作

        self.policy[state] = action

    ## 蒙特卡洛抽样产生一条经历完整的MDP序列
    def mc_sample(self):

        onesequence = []                         # 经验轨迹容器

        # 基于贪婪策略产生一条轨迹
        state = self.env.reset()
        while True:
            self.update_policy(state)             # 策略改进
            action = self.policy[state]           # 根据策略选择动作
            next_state, reward, done, _ = self.env.step(action)  # 交互
            onesequence.append((state, action, reward))          # 存储经验数据
            state = next_state
            if done:
                break
        return onesequence

    ## 蒙特卡罗每次访问策略评估一条序列
    def everyvisit_valueiter_mc(self, onesequence):
        # 访问经验轨迹中的每一个状态-动作对
        for k, data_k in enumerate(onesequence):
            state, action = data_k[0], data_k[1]       # 状态, 动作
            # 计算累积折扣奖励
            G = sum(x[2]*np.power(env.gamma, i) for
                 i, x in enumerate(onesequence[k:]))
            self.r_sum[state][action] += G            # 计算折扣奖励之和
            self.r_count[state][action] += 1                         # 计算折扣奖励次数
            self.r_Q[state][action] = self.r_sum[state][action]/self.r_count[state][action]


    ## 蒙特卡罗强化学习测试
    def mcrl(self):
        for i in range(self.num_episodes):
            # 起始探索抽样一条MDP序列
            onesequence = self.mc_sample()   # 包含了策略更新
            # 值迭代过程，结合了策略评估和策略更新(策略更新主要在mc_sample)
            self.everyvisit_valueiter_mc(onesequence)

        opt_policy = self.policy     # 最优策略
        opt_Q = self.r_Q             # 最优动作价值

        return opt_policy, opt_Q


if __name__=='__main__':
    env = blackjackEnv.BlackjackEnv()   # 导入环境模型
    env.gamma = 1.0                     # 定义折扣系数
    filepath = "./img/"
    agent = StartExplore_EveryVisit_ValueIter_MCRL(env, num_episodes=100000)
    opt_policy, opt_Q = agent.mcrl()
    for key in opt_policy.keys():
        print(key, ":", opt_policy[key], opt_Q[key])
    plt.draw(opt_policy, filepath)