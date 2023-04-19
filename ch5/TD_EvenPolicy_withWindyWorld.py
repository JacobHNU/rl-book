'''
动态规划（DP）：   每一步迭代，每一步都可以进行策略评估，
                优点： 评估效率较高，
                缺点： 需要状态转移概率模型（P(s'|s,a)）， 每次计算所有的动作值
蒙特卡罗（MC）：   需要产生一个完整回合的MDP经验轨迹（链），对每一个完整回合进行策略评估， 也就是利用一个完整回合的回报G来进行评估
                优点： 不需要状态转移概率模型，可以只计算部分动作值
                缺点： 评估效率低
时序差分（TD）：   汲取了动态规划中'bootstrap'的思想，用现有的价值估计值来更新价值估计，不需要等到回合结束
                无需状态转移概率模型。目标策略和行为策略是一致的，时序差分策略评估算法是同策略评估算法。

                # 时序差分TD
                1. bootstrap
                2. 每向前一个时间步都可以更新一次相应的动作值
                3. 免模型学习
'''
import numpy as np
from collections import defaultdict


# 平均策略
def even_policy(env, state):
    action_porob = np.ones(env.aspace_size) / env.aspace_size
    return action_porob


# 时序差分策略
def TD_actionValue(env, alpha=0.01, num_episodes=100):
    # 初始化动作价值函数q(s,a)
    Q = defaultdict(lambda: np.zeros(env.aspace_size))

    # 对每个回合执行操作
    for _ in range(num_episodes):
        # 初始化状态，动作
        state = env.reset()  # 环境状态初始化
        action_prob = even_policy(env, state)  # 根据输入策略，确定动作概率，然后随机选择确定动作
        action = np.random.choice(env.get_aspace(), p=action_prob)

        # 内部循环，回合未结束（未达到最大步数、S不是终止状态），则进行单步采样，计算动作价值估计，更新价值
        while True:
            next_state, reward, done, info = env.step(action)  # 交互一步，单步采样
            action_prob = even_policy(env, next_state)
            next_action = np.random.choice(env.get_aspace(), p=action_prob)
            Q[state][action] += alpha * (reward + env.gamma * Q[next_state][next_action] -
                                         Q[state][action])  # 时序差分，计算动作价值估计，更新价值

            if done:  # 检查是否结束
                break
            state = next_state  # 更新动作和状态
            action = next_action

    return Q


if __name__ == "__main__":
    import WindyWorld

    env = WindyWorld.WindyWorldEnv()
    alpha = 0.01
    num_episodes = 100
    Q = TD_actionValue(env, alpha, num_episodes)

    for state in env.get_sspace():
        print(state, ":", Q[state])
