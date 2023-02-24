import numpy as np
np.random.seed(0)
import scipy.optimize
import gym

# 引入环境
env = gym.make('CliffWalking-v0')
print('观测空间={}'.format(env.observation_space))
print('动作空间={}'.format(env.action_space))
print('状态数量={}, 动作数量={}'.format(env.nS, env.nA))
print('地图大小={}'.format(env.shape))

# 观测空间=Discrete(48)  48个格子
# 动作空间=Discrete(4)   [1,0,0,0] 上，[0,1,0,0]右，[0,0,1,0]下，[0,0,0,1]左
# 状态数量=48, 动作数量=4  0表示向上，     1表示向右，   2表示向下，   3表示向左
# 地图大小=(4, 12)

def play_once(env, policy):
    total_reward=0
    state = env.reset()
    while True:
        loc = np.unravel_index(state, env.shape)
        # print('状态={}, 位置={}'.format(state, loc), end=' ')
        action = np.random.choice(env.nA, p=policy[state])
        next_state, reward, done, _= env.step(action)
        # print('动作={}, 奖励={}'.format(action, reward))
        total_reward += reward
        if done:
            break
        state = next_state
    return total_reward

# 最优策略
actions = np.ones(env.shape, dtype=int)
actions[-1, :] = 0      # a[-1, :] ## 取最后一个元素里面的全部内容
actions[:, -1] = 2      # a[:,-1]  ## 取所有元素里面的的最后一个
print('actions.reshape(-1)={}'.format(actions.reshape(-1)))
optimal_policy = np.eye(4)[actions.reshape(-1)]  # np.eye(4)[actions.reshape(-1)]形成one-hot编码  # reshape(-1) 将数组变为1行
print('optimal_policy={}'.format(optimal_policy))  # 最终使得48个格子中选择的动作是都是最优的

# np.eye(4)[actions.reshape(-1)]这种操作逻辑是先将48个格子对应的动作变为一行向量，然后按照行向量里面的对应元素值给eye(4)的每一行对应的位置赋值。
# eye(4)的每一行中四个位置，值为1表示执行该位置对应的动作。
# 而actions.reshape(-1)里面的元素值表示每个动作对应的值（0表示向上，1表示向右，2表示向下，3表示向左）
# actions.reshape(-1)=[1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 2 0 0 0 0 0 0 0 0 0 0 0 2]
# 位置对应动作 [1,0,0,0] 上，[0,1,0,0]右，[0,0,1,0]下，[0,0,0,1]左
# optimal_policy=[[0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [0. 0. 1. 0.]]




total_reward = play_once(env, optimal_policy)
# print('回合奖励={}'.format(total_reward))

# 求解Bellman期望方程
def evaluate_bellman(env, policy, gamma=1.):
    a,b=np.eye(env.nS), np.zeros((env.nS))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            pi = policy[state][action]
            for p, next_state, reward, done in env.P[state][action]:
                a[state, next_state] -= (pi * gamma * p)
                b[state] +=(pi * reward *p)
    v = np.linalg.solve(a,b)   # 求解线性矩阵方程，即求解矩阵方程ax=b
    q = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for p, next_state, reward, done in env.P[state][action]:
                q[state][action] += ((reward + gamma * v[next_state])*p)  #
    return v, q

policy = np.random.uniform(size=(env.nS, env.nA))

# 随机策略的价值
policy = policy / np.sum(policy, axis=1)[:, np.newaxis]    # x[:, np.newaxis] ，放在后面，会给列上增加维度
state_values, action_values = evaluate_bellman(env, policy)
# print('状态价值={}'.format(state_values))
# print('动作价值={}'.format(action_values))


# 评估最优策略的价值
optimal_state_vlues, optimal_action_value=evaluate_bellman(env,optimal_policy)
# print('最优状态价值={}'.format(optimal_state_vlues))
# print('最优动作价值={}'.format(optimal_action_value))

def optimal_bellman(env, gamma=1.):
    p = np.zeros((env.nS, env.nA, env.nS))
    r = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for prob, next_state, reward, terminated in env.P[state][action]:
                p[state, action, next_state] += prob
                r[state, action] += (reward * prob)
    c = np.ones(env.nS)
    a_ub = gamma * p.reshape(-1, env.nS) - \
            np.repeat(np.eye(env.nS), env.nA, axis=0)
    b_ub = -r.reshape(-1)
    a_eq = np.zeros((0, env.nS))
    b_eq = np.zeros(0)
    bounds = [(None, None),] * env.nS
    res = scipy.optimize.linprog(c, a_ub, b_ub, bounds=bounds,
            method='interior-point')
    v = res.x
    q = r + gamma * np.dot(p, v)
    return v, q

optimal_state_values, optimal_action_values = optimal_bellman(env)
# print('最优状态价值 = {}'.format(optimal_state_values))
# print('最优动作价值 = {}'.format(optimal_action_values))
# optimal_actions = optimal_action_values.argmax(axis=1)
# print('最优策略 = {}'.format(optimal_actions))