'''
期望sarsa求解最优策略
    1、 初始化： q(s,a)， 如果有终止状态q(s,a)<-0
        用动作价值q确定策略\pi
    2. 时序差分更新，for _ in num_episodes:
        2.1 初始化状态， state = env.reset()
        2.2 回合未结束（未达到最大步数，S不是终止状态） While True:
            2.2.1 用动作价值q(s)确定的策略（epsilon-greedy）来确定动作A
                epsilon_greedy_policy(a) = argmax(q(s))
                action =  np.random.choice(env.action_space.nA,p=action_prob)
            2.2.2 采样   next_state, reward, done, _ = env.step(action)
            2.2.3 用期望计算回报的估计值  U <- R + gamma * \sum{\pi(a|next_state) q(next_state,a)}
            2.2.4 更新价值q(s,a)   q(s,a) += alpha*U-q(s,a)
            2.2.5 state <- next_state
'''

# 用期望sarsa算法求解windyworld问题
import numpy as np
from collections import defaultdict
import WindyWorld

# 创建一个eposjlon-gree
def create_epsilon_greedy_policy(env, Q, epsilon=0.1):
    # 内部函数
    def __policy__(state):
        nA = env.aspace_size
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_A = np.argmax(Q[state])  # 选择最优动作
        A[best_A] += 1 - epsilon
        return A

    return __policy__  # 返回epsilon_greedy函数


##  期望sarsa
def SARSA(env, num_episodes=1000, alpha=0.1, epsilon=0.1):
    nA = env.aspace_size
    # 初始化
    Q = defaultdict(lambda: np.zeros(nA))  # 动作值函数
    egreedy_policy = create_epsilon_greedy_policy(env, Q, epsilon)  # epsilon-greedy函数

    # 外层循环
    for _ in range(num_episodes):
        # 初始化状态
        state = env.reset()

        while True:
            A_prob = egreedy_policy(state)
            action = np.random.choice(np.arange(nA), p=A_prob)  # 确定动作A
            next_state, reward, done, _ = env.step(action)      # 采样
            U = reward + env.gamma * np.dot(Q[next_state], egreedy_policy(next_state))  # 用期望计算回报的估计值
            Q[state][action] += alpha*(U - Q[state][action])   # 更新价值，策略评估
            if done:         # 是否到达终止状态
                break;
            state = next_state  # 更新状态，进入下一个循环

    # 用表格表示最终策略
    P_table = np.ones((env.world_height, env.world_width))*np.inf
    for state in env.get_sspace():
        P_table[state[0]][state[1]] = np.argmax(Q[state])

    # 返回最终策略和动作值
    return P_table, Q

## 主程序
if __name__ == '__main__':
    # 构造windyworld环境
    env = WindyWorld.WindyWorldEnv()

    # 调用期望sarsa算法
    P_table, Q = SARSA(env,num_episodes=1000, alpha=0.11, epsilon=0.1)

    # 输出
    print('P=', P_table)
    for state in env.get_sspace():
        print(state, Q[state])

'''
期望sarsa算法也是和sarsa算法类似，并不能保证每次都能得到最优策略，甚至是可以到达终点的策略。
[[2. 2. 2. 3. 3. 3. 3. 3. 3. 1.]
 [1. 2. 3. 3. 3. 3. 3. 1. 1. 1.]
 [0. 3. 1. 3. 3. 3. 0. 1. 3. 1.]
 [0. 3. 3. 3. 3. 3. 3. 0. 2. 1.]
 [1. 1. 3. 3. 3. 3. 0. 1. 2. 2.]
 [1. 3. 3. 3. 3. 0. 0. 1. 1. 3.]
 [1. 3. 3. 3. 0. 0. 0. 0. 0. 2.]]
'''