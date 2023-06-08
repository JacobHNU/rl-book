'''
Q-learning中用max q(s',a)来更新动作价值，会导致“最大化偏差”，使得估计的动作价值偏大
Double Q learning用两个独立的动作价值估计值来消除偏差

Double Q-learning算法
1. 初始化 Q0，Q1  = defaultdict(lamda:np.zeros(env.aspace_size))
        Q = q0 + q1
        egreedy_policy = create_egreedy_policy(env, Q, epsilon)
2. 时序差分更新 for _ in episodes:
    2.1 初始化状态  state = env.reset()
    2.2 while True:
    2.2.1 用动作价值(q0 + q1) = q(s,.)确定的策略决定动作A
          action_prob = egreedy_policy(state)
          action = np.random.choice(np.arange(nA), p=action_prob)
    2.2.2 采样   next_state, reward, done, _ = env.step(action)
    2.2.3 随机选择更新q0或q1   以等概率选择q_i作为更新对象
    2.2.4 用改进后的策略更新回报的估计  U <- R + γ q_(1-i)(s', argmax q_i(s',a))
    2.2.5 更新动作价值   更新q_i(s,a)    q_i(s,a) <- q_i(s,a)+α(U - q_i(s,a))
    2.2.6 s <- s'
'''
# 用 Double Q-learning

import numpy as np
from collections import defaultdict
import WindyWorld
import matplotlib.pyplot as plt


def create_egreedy_policy(env, Q0, Q1, epsilon=0.1):
    def __policy__(state):
        nA = env.aspace_size
        A_prob = np.ones(nA, dtype=float) * epsilon / nA  # 平均设置每个动作概率
        best_A = np.argmax(Q0[state] + Q1[state])  # 选择最优动作
        A_prob[best_A] += 1 - epsilon  # 设定贪婪动作概率
        return A_prob

    return __policy__  # 返回epsilon-greedy策略


# Double Q-learning
def Double_Q_learning(env, epsilon=0.1, alpha=0.1, num_episodes=1000):
    nA = env.aspace_size
    Q = defaultdict(lambda: np.zeros(nA))
    Q0 = defaultdict(lambda: np.zeros(nA))
    Q1 = defaultdict(lambda: np.zeros(nA))

    egreedy_policy = create_egreedy_policy(env, Q0, Q1, epsilon)  # 贪婪策略函数

    episodes_rewards = []
    # 外层循环
    for _ in range(num_episodes):
        episode_reward = 0
        state = env.reset()  # 状态初始化
        # 内层循环
        while True:
            # 用动作价值(q0 + q1) = q(s,.)确定的策略决定动作A
            action_prob = egreedy_policy(state)  # 产生当前动作
            action = np.random.choice(np.arange(nA), p=action_prob)
            # 采样
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            # 以等概率选择q_i作为更新对象
            if np.random.rand() >= 0.5:
                Q0_max_a = np.argmax(Q0[next_state])
                U0 = reward + env.gamma * Q1[next_state][Q0_max_a]
                Q0[state][action] += alpha * (U0 - Q0[state][action])
            else:
                Q1_max_a = np.argmax(Q1[next_state])
                U1 = reward + env.gamma * Q0[next_state][Q1_max_a]
                Q1[state][action] += alpha * (U1 - Q1[state][action])
            if done:
                break

            state = next_state
        episodes_rewards.append(episode_reward)
    # 用表格表示最终策略
    P_table = np.ones((env.world_height, env.world_width)) * np.inf
    for state in env.get_sspace():
        P_table[state[0]][state[1]] = np.argmax(Q0[state])

    # 返回最终策略和动作值
    return P_table, Q, episodes_rewards


# 主程序
if __name__ == '__main__':
    env = WindyWorld.WindyWorldEnv()

    P_table, Q, episodes_rewards = Double_Q_learning(env, epsilon=0.1, alpha=0.1, num_episodes=9000)

    print('P=', P_table)
    for state in env.get_sspace():
        print(state, Q[state])
    plt.plot(episodes_rewards)
    plt.show()

'''
[[3. 1. 3. 3. 3. 3. 3. 3. 3. 1.]
 [1. 0. 3. 3. 3. 3. 2. 0. 0. 1.]
 [1. 2. 3. 3. 3. 1. 2. 0. 1. 1.]
 [3. 3. 3. 3. 2. 1. 2. 0. 2. 1.]
 [3. 3. 1. 0. 1. 0. 0. 1. 2. 2.]
 [0. 2. 2. 3. 0. 0. 0. 3. 2. 3.]
 [0. 1. 0. 2. 0. 0. 0. 0. 3. 0.]]
'''
