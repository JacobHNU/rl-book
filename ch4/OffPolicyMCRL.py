### 异策略：一个用于决策的behavior policy \miu ,  一个用于更新的target policy \pi，
##  当\miu 和 \pi一样时就是on-policy, 不一样是就是off-policy。
### 异策略蒙特卡罗更新，为什么要采用重要性采样？  https://zhuanlan.zhihu.com/p/371156865
## 重要性采样出现的原因是原始分布难以直接采样，故需要借助一个简单、可采样的分布来计算期望。
# 但在强化学习中使用重要性采样不是因为原始分布难以采样，而是不想通过这个分布进行采样。
# 重要性采样的的主要用处在于：用在两种策略下观察到的动作的概率的比值对回报进行加权，从而把行动策略下的期望值转化为目标策略下的期望值。

'''
异策略蒙特卡罗增量式每次访问策略评估和改进
'''

from collections import defaultdict
import numpy as np
import blackjackEnv
import plot_func as plt

class OffPolicyMCRL_ImportantSampling:
    # 类初始化： 动作价值函数q(s,a), 权重和W_sum，目标策略tgt_policy，行为策略act_policy
    def __init__(self, env, num_episodes=100000):
        self.env = env
        self.nA = env.action_space.n  # 动作空间维度
        self.r_Q = defaultdict(lambda: np.zeros(self.nA))  # 动作值函数
        self.W_sum = defaultdict(lambda: np.zeros(self.nA))  # 累积重要性权重
        self.tgt_policy = defaultdict(lambda: np.zeros(self.nA))  # 目标策略
        self.act_policy = defaultdict(lambda: np.zeros(self.nA))  # 行为策略
        self.num_episodes = num_episodes  # 最大抽样次数

    # 初始化目标策略及更新目标策略
    def target_policy(self, state):
        if state not in self.tgt_policy.keys():
            player, dealer, ace = state
            action = 0 if player >= 18 else 1
        else:
            action = np.argmax(self.r_Q[state])  # 最优动作值对应的动作
            self.tgt_policy[state] = np.eye(self.nA)[action]  # one-hot编码 [a（要牌）,b（不要牌）]:  a，b默认为0，为1则表示动作生效。
        return self.tgt_policy[state]

    # 初始化行为策略（行为策略不更新）
    def action_policy(self, state):
        self.act_policy[state] = [0.5, 0.5]
        return self.act_policy[state]

    ## 按照行为策略蒙特卡罗抽样产生一条经历完整的MDP序列
    def mc_sample(self):
        onesequence = []           # 经验轨迹容器
        state = self.env.reset()   # 初始状态
        while True:
            action_prob = self.action_policy(state)
            # 以行为策略的概率选取动作
            action = np.random.choice(np.arange(len(action_prob)),p=action_prob)    # p实际是个数组，大小（size）应该与指定的action_space,nA相同，用来规定选取a中每个元素的概率
            next_state, reward, done, _ = self.env.step(action)   # 交互一步
            onesequence.append((state, action, reward))
            if done:   # 游戏是否结束
                break
            state = next_state
        return onesequence

    # 基于值迭代的增量式异策略蒙特卡罗每次访问策略评估和改进
    def offpolicy_everyvisit_mc_valueiter(self, onesequence):
        G = 0.    # 初始化回报
        W = 1     # 初始化权重
        # 自后向前依次遍历MDP序列中的所有状态-动作对
        for j in range(len(onesequence)-1,-1,-1):   # range(start, end, step), [start, end), 因此倒序的话需要len(onesequence)-1才是最后一个， step=-1，表示往前遍历
            state, action = onesequence[j][0], onesequence[j][1]
            # 计算折扣奖励
            G = G + env.gamma * onesequence[j][2]   # 累积折扣奖励
            # 计算重要性权重
            W = W + (self.target_policy(state)[action]/self.action_policy(state)[action])   # 重要性权重   W = W * (target_policy(state)[action])/(action_policy(state)[action])
            if W == 0:    # 权重为0，需要退出循环避免除0错误
                break;
            self.W_sum[state][action] += W           # 权重之和
            # 更新价值
            self.r_Q[state][action] += (G - self.r_Q[state][action])*W/self.W_sum[state][action]    # q = q + W/W_sum * (G-q)
            # 策略改进
            self.target_policy(state)

    def mcrl(self):
        for i in range(self.num_episodes):
            # 用行为策略抽样一条MDP轨迹
            onesequence = self.mc_sample()
            # 蒙特卡罗策略评估和目标策略改进
            self.offpolicy_everyvisit_mc_valueiter(onesequence)

        return self.tgt_policy, self.r_Q

# 测试
if __name__ == "__main__":
    env  = blackjackEnv.BlackjackEnv()   # 导入环境模型
    env.gamma = 1.                       # 定义折扣系数
    filepath = "./img/"

    # 定义方法
    agent = OffPolicyMCRL_ImportantSampling(env)
    # 强化学习
    opt_tgt_policy, opt_r_Q = agent.mcrl()
    # 打印结果
    for key in opt_tgt_policy.keys():
        print(key, ":", opt_tgt_policy[key], opt_r_Q[key])
    plt.draw3(opt_tgt_policy, filepath)
