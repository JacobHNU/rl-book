### 首次访问蒙特卡罗策略评估，计算动作值函数
# 简单策略为：如果player手中牌的点数大于等于18，则停止要牌，否则继续要牌。
###
import numpy as np
import blackjackEnv
from collections import defaultdict



# 待评估的玩家策略； 点数小于18则继续叫牌，否则停牌
def simple_policy(state):
    player, dealer, ace = state
    return 0 if player >= 18 else 1  # 0:停牌 ， 1：要牌


# 首次访问MC策略评估
def firstvisit_mc_actionvalue(env, num_episode=50000):

    # 初始化动作价值r_Q, 更新动作价值时的计数器r_count, 奖励和r_sum
    r_sum = defaultdict(float)    # 记录状态-动作对的累积折扣奖励之和
    r_count = defaultdict(float)  # 记录状态-动作对的累积折扣奖励次数
    r_Q = defaultdict(float)      #  动作值样本均值

    # reward = 0.
    # 采样num_episodes条经验轨迹
    MDPsequence = []    # 经验轨迹容器
    for i in range(num_episode):
        state = env.reset()   # 环境状态初始化
        while True:
            # 采集一条经验轨迹
            onesequence = []
            # （采样） 用策略simple_policy生成轨迹
            action = simple_policy(state)
            next_state, reward, done, info = env.step(action)  # 交互一步
            # （统计首次出现的步骤）
            onesequence.append((state, action, reward))  # MDP序列
            if done:  # 游戏是否结束
                break
            state = next_state
        MDPsequence.append(onesequence)
    # 计算动作值，即策略评估
    for i in range(len(MDPsequence)):
        onesequence = MDPsequence[i]
        # 更新回报
        SA_pairs = []
        for j in range(len(onesequence)):
            sa_pair = (onesequence[j][0], onesequence[j][1])
            if sa_pair not in SA_pairs:
                SA_pairs.append(sa_pair)
                G = sum([x[2]*np.power(env.gamma, k) for
                         k, x in enumerate(onesequence[j:])])
                r_sum[sa_pair] += G  # 合并折扣奖励
                r_count[sa_pair] += 1  # 记录次数
    for key in r_sum.keys():
        r_Q[key] = r_sum[key]/r_count[key]  # 计算样本均值

    return r_Q, r_count



# 主程序入口
if __name__ == '__main__':
    env = blackjackEnv.BlackjackEnv()  # 定义环境模型
    env.gamma = 1.0                    # 补充定义折扣系数
    r_Q, r_count = firstvisit_mc_actionvalue(env)  # 调用主函数
    # 打印结果
    for key, data in r_Q.items():
        print(key,r_count[key],':', data)
