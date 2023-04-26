'''
1. n 步时序差分法，至少n步交互后的经验数据才能计算TD目标值
2. 在交互达到终止状态后，后n个动作值已经不能再使用n步TD目标值进行更新了，因为此时已不足n步。

时序差分 TD Target 约等于 G(s_t+1, a_t+1)
3. n 步时序差分法，介于单步TD法和蒙特卡洛法之间，
    当n = 1 时， 则为单步时序差分  G(s_t, a_t) = R_t+1 + gamma* Q(s_t+1, a_t+1)
    当n = T时， 则为蒙特卡洛法（整个回合的抽样） G(s_t, a_t) = R_t+1 + gamma*R_t+2 + ... + gamma^T* Q(s_T, a)    Q(s_T, a)恒等于0

'''

# n步时序差分策略评估算法

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# 平均策略
def even_policy(env):
    nA = env.aspace_size
    action_prob = np.ones(nA)/nA
    return action_prob


# 时序差分策略评估
def nstep_TD_policyEstimate(env, nstep, alpha=0.1,delta=0.1):
    nA = env.aspace_size
    aspace = env.get_aspace()
    Q = defaultdict(lambda: np.zeros(nA))
    error = 0.0                       # 前后两次动作值最大差值
    episodes_rewards = []
    # 外层循环直到动作改变小于容忍系数
    while error <= delta:
        # 环境状态初始化
        state = env.reset()
        nstep_mdp = []                 # 存储n步交互数据
        # 利用既定策略选择动作
        action_prob = even_policy(env)
        action = np.random.choice(aspace, p=action_prob)
        episodes_reward = 0
        # 内层循环，直到到达终止状态
        while True:
            # 采样
            next_state, reward, end, info = env.step(action)
            nstep_mdp.append((state, action, reward))  # 生成n步
            episodes_reward += reward

            # 未到n步就已到达终止状态，则直接退出
            if len(nstep_mdp) < nstep:
                if end == True:
                    break
                else:
                    # 选择动作
                    action_prob = even_policy(env)
                    next_action = np.random.choice(aspace, p=action_prob)
            if len(nstep_mdp) >= nstep:
                if end == False:
                    # 根据平均策略选择一个动作
                    action_prob = even_policy(env)
                    next_action = np.random.choice(aspace, p=action_prob)

                    # 之前第n步的动作和状态
                    state_n, action_n = nstep_mdp[0][0], nstep_mdp[0][1]    # n步操作的开始
                    Q_temp = Q[state_n][action_n]

                    # 计算n步TD目标值G
                    Re = [x[2] for x in nstep_mdp]
                    Re_sum = sum([env.gamma**i*re for (i,re) in enumerate(Re)])  # U = R_t+1 + γR_t+2 + ... +γ^(n-1)*R_t+n
                    G = Re_sum+env.gamma**nstep*Q[next_state][next_action]  # G = U + γ^n * Q[s_t+n, a_t+n]  此处是Q[next_state][next_action]

                    # n步时序差分价值更新
                    Q[state_n][action_n] += alpha*(G-Q[state_n][action_n])

                    # 更新最大误差
                    error = max(error, abs(Q[state_n][action_n]-Q_temp))

                    # 删除n步片段中最早一条交互数据
                    nstep_mdp.pop(0)    # 删除n步片段的第一条，开始下一轮的n步，则从第二条交互数据开始。

                    episodes_rewards.append(episodes_reward)
                else:   # 已到达终止状态，处理剩下不足n步的交互数据,不足n步的就按照当前步数的来处理。
                    for i in range(len(nstep_mdp)):
                        state_i = nstep_mdp[i][0]     # 状态
                        action_i = nstep_mdp[i][1]    # 动作
                        Q_temp = Q[state_i][action_i]  # 临时保存旧值

                        # 计算剩下部分TD目标值G
                        Re = [x[2] for x in nstep_mdp[i:]]
                        G = sum(env.gamma**i*re for (i, re) in enumerate(Re))

                        # 时序差分更新
                        Q[state_i][action_i] += alpha*(G-Q[state_i][action_i])

                        # 更新最大误差
                        error = max(error, abs(Q[state_i][action_i] - Q_temp))
                        episodes_rewards.append(episodes_reward)
                    break   # 本轮循环结束

            state = next_state
            action = next_action

    # 用表格表示最终策略
    P_table = np.ones((env.world_height, env.world_width)) * np.inf
    for state in env.get_sspace():
        P_table[state[0]][state[1]] = np.argmax(Q[state])

    return P_table, Q, episodes_rewards

if __name__ == '__main__':
    import WindyWorld
    env = WindyWorld.WindyWorldEnv()

    nstep_list = [3, 6, 10, 15]
    episodes_reward_rec = [[] for i in range(len(nstep_list))]

    for nstpe_idx, nstep in enumerate(nstep_list):
        episodes_rewards = []
        P_table, Q, episodes_rewards = nstep_TD_policyEstimate(env, nstep, alpha=0.1, delta=0.1)
        episodes_reward_rec[nstpe_idx] = episodes_rewards
        print('P{}={}'.format(nstep_list[nstpe_idx],P_table))
        # for state in env.get_sspace():
        #     print(state, Q[state])

    for idx, ep_reward in enumerate(episodes_reward_rec):
        plt.plot(range(len(ep_reward)), ep_reward, label='nstep={}'.format(nstep_list[idx]))

    plt.xlabel("Episodes")
    plt.ylabel("Episodes_rewards")
    plt.legend()
    plt.show()

