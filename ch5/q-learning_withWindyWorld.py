"""
基于重要性采样的异策算法
异策略时序差分
                               同策                                    异策略
TD target :               R + gamma * Q(s',a')                  R + gamma * max(Q(s', a))
采样 a', s'       计算TD target，利用相同策略得到a'，计算Q(s',a')     利用π采样a，计算TD Target是用改进后策略，max()计算Q(s',a)

SARSA算法计算TD target: U=R + gamma * q(s',a')               q(s,a)表示在某个状态动作对(s,a)下的期望回报
期望SARSA算法计算TD target： U = R + gamma *　v(s')            v(s) = π(a|s)Q(s,a)，换成了v(s)则是针对状态s的期望回报，从而避免了偶尔出现的不正当行为给整体结果带来的负面影响
Q-Learning算法计算TD target： U = R + gamma * max(q(s',a))    Q学习算法认为在根据s’估计TD target时，与其使用q(s',a')或v(s')，不如使用根据q(s',.)改进后的策略来更新，这样可以更接近最优价值。
"""

'''
Q-Learning算法流程
1. 初始化q(s,a)
2. 时序差分更新， for _ in range(num_episodes):
    2.1 初始化状态  state = env.reset()
    2.2 如果回合未结束（未达到最大步数，S不是终止状态） while True:
        2.2.1 用动作价值估计q(s,.)确定策略π决定动作action （如epsilon-Greedy策略） 
              action_prob = egreedy_policy(state)
              action = np.random.choice(np.arange(NA), p=action_prob)
        2.2.2 采样交互数据  next_state, reward, done, _ = env.step(action)
        2.2.3 用改进后的策略计算回报 U=np.max(q[next_state])
        2.2.4 更新价值和策略   q[state][action] += alpha * (U - q[state][action])
        2.2.5 state = next_state 
'''
import numpy as np
from collections import defaultdict
import WindyWorld


def create_egreedy_policy(env, Q, epsilon=0.1):
    def __policy__(state):
        nA = env.aspace_size
        A_prob = np.ones(nA, dtype=float) * epsilon / nA  # 平均设置每个动作概率
        best_A = np.argmax(Q[state])  # 选择最优动作
        A_prob[best_A] += 1 - epsilon  # 设定贪婪动作概率
        return A_prob

    return __policy__


def Q_Learning(env, alpha=0.1, epsilon=0.1, num_episodes=1000):
    nA = env.aspace_size
    # 初始化Q(s,a)
    Q = defaultdict(lambda: np.zeros(nA))
    egreedy_policy = create_egreedy_policy(env, Q, epsilon)

    for _ in range(num_episodes):
        # 初始化状态
        state = env.reset()

        while True:
            # 用动作价值估计q(s,.)确定策略π决定动作action
            action_prob = egreedy_policy(state)  # 用贪心策略确定动作概率
            action = np.random.choice(np.arange(nA), p=action_prob)  # 确定动作
            # 采样交互数据
            next_state, reward, done, _ = env.step(action)
            # 用改进后的策略计算回报
            U = reward + env.gamma * np.max(Q[next_state])
            # 更新价值和策略
            Q[state][action] += alpha * (U - Q[state][action])
            # 判断是否终止
            if done:
                break
            # 更新状态进行下一个循环
            state = next_state
    # 用表格表示最终策略
    P_table = np.ones((env.world_height, env.world_width)) * np.inf
    for state in env.get_sspace():
        P_table[state[0]][state[1]] = np.argmax(Q[state])

    # 返回最终策略和动作值
    return P_table, Q


# main
if __name__ == '__main__':
    # 构造环境
    env = WindyWorld.WindyWorldEnv()

    P_table, Q = Q_Learning(env, alpha=0.1, epsilon=0.1, num_episodes=10000)

    # 打印
    print('P=', P_table)
    for state in env.get_sspace():
        print(state,Q[state])

'''
生成能够从起点（3，0）到终点（3,7）的策略表，也是有概率的，并不是每1000，10000或者100000个回合都能成功生成。
P=[[0. 2. 3. 3. 3. 3. 3. 3. 3. 1.]
 [2. 3. 3. 3. 3. 3. 3. 3. 1. 1.]
 [3. 3. 3. 3. 3. 3. 3. 0. 3. 1.]
 [3. 3. 3. 3. 3. 1. 3. 0. 2. 1.]
 [1. 1. 3. 3. 3. 1. 0. 1. 2. 2.]
 [1. 3. 3. 3. 1. 0. 0. 1. 1. 1.]
 [3. 3. 3. 0. 0. 0. 0. 0. 0. 2.]]
'''