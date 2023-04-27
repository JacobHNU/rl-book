'''
   2023/04/14
   auther: Jacob
  同策时序差分更新来求解最优策略
  sarsa: on-policy
  算法描述：
        输入：环境（env，无数学描述）
        输出：最优策略估计\pi(a|s) 和最优动作价值估计q(s,a)
        参数： 学习率alpha，折扣因子gamma，策略改进的参数epsilon，回合数num_episodes
        1.(初始化): q(s,a)<-defaultdict(), q(s终止,a)<-0
        2.(时序差分更新): for _ in num_episodes:
            2.1 初始化状态、动作， 选择状态s，state = env.reset()  用策略\pi确定动作a,  action = policy(state)
            2.2  while True:  回合未结束（未到达最大步数，s不是终止状态等）
                2.2.1 (采样)  next_state, reward, done, info = env.step(action)
                2.2.2 (确定next_action) next_action = policy(next_state)
                2.2.3 (计算动作价值估计，并更新价值)
                 Q(state, action) += alpha*(reward+gamma*Q(next_state,next_action) - Q(state,action))
                2.2.4 (更新策略)  policy(state, action) = argmax(Q(state,action))
                2.2.5 s<-s', a<-a'
'''

# 用sarsa算法求解windyworld问题
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# 创建一个epsilon-greedy策略
def epsilon_greedy_Policy(env, Q, epsilon=0.1):
    # 内部函数
    def __policy__(state):
        NA = env.aspace_size
        A = np.ones(NA, dtype=float) * epsilon / NA  # 平均设置每个动作概率
        best = np.argmax(Q[state])  # 选择最优动作
        A[best] += 1 - epsilon  # 设定贪婪动作概率
        return A

    return __policy__  # 返回epsilon-greedy策略函数


## Sarsa
def SARSA(env, num_episodes=500, alpha=0.1, epsilon=0.1):
    nA = env.aspace_size
    episodes_reward = []

    # 初始化
    Q = defaultdict(lambda: np.zeros(nA))  # 动作值
    egreedy_policy = epsilon_greedy_Policy(env, Q, epsilon)  # 贪婪策略函数

    # 外层循环
    for _ in range(num_episodes):
        state = env.reset()  # 初始化状态
        episode_reward = 0.
        # 内层循环
        while True:
            # 用贪婪策略选择动作
            action_prob = egreedy_policy(state)
            action = np.random.choice(np.arange(nA), p=action_prob)
            # 采样
            next_state, reward, done, info = env.step(action)  # 采样，交互一步
            episode_reward += reward
            action_prob = egreedy_policy(next_state)
            # 确定next_action
            next_action = np.random.choice(np.arange(nA), p=action_prob)
            # (计算动作价值估计，并更新价值)
            Q[state][action] += alpha * (reward
                                         + env.gamma * Q[next_state][next_action] - Q[state][action])
            # 到达最终状态，退出本回合
            if done:
                break

            state = next_state
        episodes_reward.append(episode_reward)
    # 用表格表示最终策略
    P_table = np.ones((env.world_height, env.world_width)) * np.inf
    for state in env.get_sspace():
        P_table[state[0]][state[1]] = np.argmax(Q[state])

    # 返回最终策略和动作值
    return P_table, Q, episodes_reward


if __name__ == "__main__":
    # 构造WindyWorld环境
    import WindyWorld

    env = WindyWorld.WindyWorldEnv()
    num_episodes = 1000
    # 调用sarsa
    P_table, Q, episodes_reward = SARSA(env, num_episodes, alpha=0.1, epsilon=0.1)

    # 输出
    print('p=', P_table)
    for state in env.get_sspace():
        print(state, Q[state])

    plt.plot(episodes_reward)
    plt.show()

'''
输出有个问题，就是动作状态的不确定，有的时候并不能顺利到达终点
alpha=0.01, epsilon=0.01 时很难到达终点，并且奖励曲线一路下降到-100000，
随着alpha的增大回合的奖励值逐步增加，在0.08后稳定在-35000，
同时还需要epsilon=0.08,alpha=0.1时到达终点的Q表的正确率才保持增加

alpha=0.01, epsilon=0.01
p= [[0. 3. 3. 1. 3. 1. 3. 3. 1. 3.]
 [2. 3. 1. 3. 3. 3. 1. 1. 3. 1.]
 [3. 3. 3. 3. 3. 3. 3. 1. 1. 1.]
 [3. 3. 3. 3. 3. 2. 3. 0. 3. 1.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
alpha=0.1, epsilon=0.1
 [[2. 3. 1. 1. 3. 3. 3. 3. 3. 1.]
 [3. 3. 3. 1. 3. 3. 3. 0. 2. 1.]
 [3. 3. 3. 3. 3. 3. 0. 1. 3. 1.]
 [3. 3. 3. 3. 3. 1. 0. 0. 2. 1.]
 [3. 1. 3. 3. 3. 3. 0. 1. 2. 2.]
 [3. 1. 3. 3. 3. 0. 0. 1. 1. 2.]
 [3. 3. 3. 3. 0. 0. 0. 0. 0. 0.]]
'''