## 柔性策略可以选择所有可能的动作，所以从一个状态出发可以达到这个状态能达到的所有状态和所有状态动作对。
## 采用柔性策略，有助于全面覆盖所有的状态或状态动作对，这样就能求得全局最优策略。

# 初始化:  动作价值函数，计数器，柔性策略    class 类名
# 策略初始化，改进策略方法 update_policy()
# 利用epsilon-贪婪策略采样一条完整轨迹 mc_sample()，
# 对轨迹进行策略评估

from collections import defaultdict
import numpy as np
import blackjackEnv
import plot_func as plt

'''
基于值迭代的epsilon-贪婪策略每次访问蒙特卡罗强化学习算法
'''


class SoftExplore_EveryVisit_ValueIter_MCRL:
    # 初始化 动作价值函数q(s,a)，计数器c(s,a),柔性策略\pi
    def __init__(self, env, num_episodes=10000, epsilon=0.1):
        self.env = env
        self.nA = env.action_space.n  # 动作空间维度
        self.r_Q = defaultdict(lambda: np.zeros(self.nA))  # 动作值函数
        self.r_sum = defaultdict(lambda: np.zeros(self.nA))  # 累积折扣奖励之和
        self.r_count = defaultdict(lambda: np.zeros(self.nA))  # 累积折扣奖励次数
        self.greedy_policy = defaultdict(int)  # 贪婪策略
        self.eg_policy = defaultdict(lambda: np.zeros(self.nA))  # epsilon贪婪策略
        self.num_episodes = num_episodes  # 最大抽样次数
        self.epsilon = epsilon

    ## 策略初始化及其改进函数，初始化为：点数小于18继续叫牌，否则停牌
    def update_policy(self, state):
        if state not in self.greedy_policy.keys():
            player, dealer, ace = state
            action = 0 if player >= 18 else 1  # 0停牌，1：要牌
        else:
            action = np.argmax(self.r_Q[state])  # 最优价值对应的动作

        # 贪婪策略
        self.greedy_policy[state] = action
        # 对应的epsilon贪婪策略
        self.eg_policy[state] = np.ones(self.nA) * self.epsilon / self.nA   #
        self.eg_policy[state][action] += 1 - self.epsilon

        return self.greedy_policy[state], self.eg_policy[state]

    ## 蒙特卡罗采样产生一条经历完整的MDP序列
    def mc_sample(self):
        onesequence = []  # 经验轨迹容器

        # 基于epsilon-贪婪策略产生一条轨迹
        state = self.env.reset()  # 初始状态
        while True:
            _, action_prob = self.update_policy(state)
            action = np.random.choice(np.arange(len(action_prob)),
                                      p=action_prob)  # 柔性策略可以选择所有可能的动作
            next_state, reward, done, _ = self.env.step(action)
            onesequence.append((state, action, reward))
            state = next_state
            if done:
                break
        return onesequence

    ## 蒙特卡罗每次访问策略评估一条序列
    def everyvisit_valueiter_mc(self, onesequence):
        # 访问经验轨迹中的每一条状态-动作对
        for k, data_k in enumerate(onesequence):
            state = data_k[0]
            action = data_k[1]
            # 计算累积折扣奖励
            G = sum(x[2] * np.power(env.gamma, i) for i, x
                    in enumerate(onesequence[k:]))
            self.r_sum[state][action] += G  # 累积折扣奖励之和
            self.r_count[state][action] += 1.0  # 累积折扣奖励次数
            self.r_Q[state][action] = self.r_sum[
                                          state][action] / self.r_count[state][action]

    ## 蒙特卡罗测试
    def mcrl(self):
        for i in range(self.num_episodes):
            # 起始探索抽样一条MDP序列
            onesequence = self.mc_sample()

            # 值迭代过程，结合了策略评估和策略改进
            self.everyvisit_valueiter_mc(onesequence)

        opt_policy = self.greedy_policy
        opt_Q = self.r_Q

        return opt_policy, opt_Q


if __name__ == "__main__":
    env = blackjackEnv.BlackjackEnv()  # 导入环境模型
    env.gamma = 1.0  # 定义折扣系数
    filepath = "./img/"

    agent = SoftExplore_EveryVisit_ValueIter_MCRL(env, num_episodes=10000, epsilon=0.1)
    opt_policy, opt_Q = agent.mcrl()
    for key in opt_policy.keys():
        print(key, ":", opt_policy[key], opt_Q[key])
    plt.draw(opt_policy, filepath)
