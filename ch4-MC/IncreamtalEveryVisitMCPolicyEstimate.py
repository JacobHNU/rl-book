# 增量式每次访问蒙特卡罗策略评估，计算动作价值函数
# 策略还是简单策略

import numpy as np
from collections import defaultdict
import blackjackEnv

# 待评估的玩家策略； 点数小于18则继续叫牌，否则停牌
def simple_policy(state):
    player, dealer, ace = state
    return 0 if player >= 18 else 1  # 0:停牌 ， 1：要牌

'''
  增量式每次访问蒙特卡罗策略评估
'''
def everyvisit_incremental_mc_actionvalue(env, num_episodes=50000):
   r_count = defaultdict(float)
   r_Q = defaultdict(float)

    # 1. 逐次采样
   for i in range(num_episodes):
        #    采样一条经验轨迹
        state = env.reset()
        onesequence = []
        while True:
            action = simple_policy(state)  # 根据给定的简单策略选择动作
            next_state, reward, done, _ = env.step(action)  # 交互一步
            onesequence.append((state, action, reward))  # MDP序列
            if done:
                break
            state = next_state

        # 2. 逐个更新动作值样本均值
        for j in range(len(onesequence)):
            sa_pair = (onesequence[j][0],onesequence[j][1])
            G = sum([x[2]*np.power(env.gamma, k) for
                     k, x in enumerate(onesequence[j:])])
            r_count[sa_pair] += 1
            r_Q[sa_pair] += (1.0/r_count[sa_pair]) * (G-r_Q[sa_pair])

   return r_Q, r_count

if __name__ == '__main__':
    env = blackjackEnv.BlackjackEnv()  # 定义环境模型
    env.gamma = 1.0                    # 定义折扣函数

    r_Q, r_count = everyvisit_incremental_mc_actionvalue(env)
    # 打印结果
    for key, data in r_Q.items():
        print(key, r_count[key], ":", data)
