## 1、上一章的bellman期望方程和最优方程转化为标准形式用 sympy.Matrix来求还是可以较好理解，但是可以看出直接求解有点复杂。
## 2、最优策略
# actions = np.ones(env.shape, dtype=int)
# actions[-1, :] = 0      # a[-1, :] ## 取最后一个元素里面的全部内容
# actions[:, -1] = 2      # a[:,-1]  ## 取所有元素里面的的最后一个
# # print('actions.reshape(-1)={}'.format(actions.reshape(-1)))
# optimal_policy = np.eye(4)[actions.reshape(-1)]  # np.eye(4)[actions.reshape(-1)]形成one-hot编码  # reshape(-1) 将数组变为1行
# print('optimal_policy={}'.format(optimal_policy))  # 最终使得48个格子中选择的动作是都是最优的
## 最优策略的设置就是将每个格子（状态）对应的动作赋予最优的动作，使得可以最短最快到达终点。

# ch3
# Bellman算子是压缩映射，因此可以用Banach不动点定理迭代求解Bellman方程。（不懂）
# 有模型策略迭代，即在给定的动力系统矩阵p（这就是模型）

# 策略评估-> 得到状态价值v(s) -> 策略迭代-> 得到动作价值q(s,a), 以及argmax q(s,a)-> 更新策略\pi -> 策略评估...

# 冰面滑行
import numpy as np

np.random.seed(0)
import gym

env = gym.make('FrozenLake-v1')
print('观察空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测空间大小 = {}'.format(env.observation_space.n))
print('动作空间大小 = {}'.format(env.action_space.n))
transition_p = env.unwrapped.P[14][2]  # 只用gym.make('FrozenLake-v1')得到的是一个经过包装的环境，而用env.unwrapped则可以得到原始的类
print(transition_p)


# 观察空间 = Discrete(16)
# 动作空间 = Discrete(4)  0左，1下，2右，3上
# 观测空间大小 = 16    {s0,...,s15}
# 动作空间大小 = 4
# transition_p=
## [(0.3333333333333333, 14, 0.0, False),  （概率，下一状态，奖励值，回合结束标志）
# (0.3333333333333333, 15, 1.0, True),
# (0.3333333333333333, 10, 0.0, False)]
## 一开始随机策略下到下一个状态的概率都是相等的

######## 随机策略
def play_policy(env, policy, render=True):
    total_reward = 0.
    observation = env.reset()
    while True:
        if render:
            env.render()  # 显示环境
        action = np.random.choice(env.action_space.n, p=policy[observation])
        observation, reward, done, _ = env.step(action)
        total_reward += reward  # 统计回合奖励
        if done:  # 游戏结束
            break
    return total_reward


# 随机策略
random_policy = \
    np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
# print("随机策略:{}".format(random_policy))
episode_rewards = [play_policy(env, random_policy) for _ in range(1000)]
print("随机策略 平均奖励: {}".format(np.mean(episode_rewards)))


########## 策略评估
def v2q(env, v, s=None, gamma=1.):  # 根据状态价值函数计算动作价值函数
    if s is not None:  # 针对单个状态求解
        q = np.zeros(env.action_space.n)  # 用env.unwrapped可以得到原始的类
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                # q=\sum_{s',r} prob(s',r|s,a) * [r + \gamma v_\pi(s')]
                q[a] += prob * (reward + gamma * v[next_state] * (1. - done))  # 为什么要乘以(1. - done)这个不太懂
    else:  # 针对所有状态求解
        q = np.zeros((env.observation_space.n, env.action_space.n))
        for s in range(env.observation_space.n):
            q[s] = v2q(env, v, s, gamma)  # 递归
    return q


def evaluate_policy(env, policy, gamma=1., tolerant=1e-6):
    v = np.zeros(env.observation_space.n)
    while True:  # 循环
        delta = 0
        for s in range(env.observation_space.n):
            vs = sum(policy[s] * v2q(env, v, s, gamma))  # 更新状态价值函数  v(s) = \sum_{a} \pi(a|s) * q(s,a)
            delta = max(delta, abs(v[s] - vs))  # 更新最大误差
            v[s] = vs  # 更新状态价值函数
        if delta < tolerant:  # 查看是否满足迭代条件
            break;
    return v


# 对随机策略进行评估 -> 得到状态价值函数
print("状态价值函数:")
v_random = evaluate_policy(env, random_policy)
print(v_random.reshape(4, 4))
# 状态价值函数:
# [[0.0139372  0.01162942 0.02095187 0.01047569]
#  [0.01624741 0.         0.04075119 0.        ]
#  [0.03480561 0.08816967 0.14205297 0.        ]
#  [0.         0.17582021 0.43929104 0.        ]]

print("动作价值函数")
q_random = v2q(env, v_random)
print(q_random)


# 动作价值函数
# [[0.01470727 0.01393801 0.01393801 0.01316794]
#  [0.00852221 0.01162969 0.01086043 0.01550616]
#  [0.02444416 0.0209521  0.02405958 0.01435233]
#  [0.01047585 0.01047585 0.00698379 0.01396775]
#  [0.02166341 0.01701767 0.0162476  0.01006154]
#  [0.         0.         0.         0.        ]
#  [0.05433495 0.04735099 0.05433495 0.00698396]
#  [0.         0.         0.         0.        ]
#  [0.01701767 0.04099176 0.03480569 0.04640756]
#  [0.0702086  0.11755959 0.10595772 0.05895286]
#  [0.18940397 0.17582024 0.16001408 0.04297362]
#  [0.         0.         0.         0.        ]
#  [0.         0.         0.         0.        ]
#  [0.08799662 0.20503708 0.23442697 0.17582024]
#  [0.25238807 0.53837042 0.52711467 0.43929106]
#  [0.         0.         0.         0.        ]]

######### 策略改进
def improve_policy(env, v, policy, gamma=1.):
    optimal = True
    for s in range(env.unwrapped.nS):
        q = v2q(env, v, s, gamma)  # 得到q(s,a)
        a = np.argmax(q)  # 找到使得q(s,a)最大的动作 a' = argmax_a q(s,a)
        if policy[s][a] != 1.:  # 用1表示当前动作被选用，policy[s][a]表示当前状态s下的动作a 是否=1就表明当前状态s下动作a是否被选，
            optimal = False  # 如果没有被选则不是最优(因为这里的a是argmax(q)得到的)
            policy[s] = 0.  # 当前的状态下的最优动作未被选择，因此将该状态下的动作都赋0，然后再选择该状态下的最优动作，
            policy[s][a] = 1.  # 将当前状态下的最优动作[s][a]位置赋1，表示最优策略，这样策略就改进了
    return optimal


# 对随机策略进行改进
policy = random_policy.copy()  # 防止原始策略被修改
optimal = improve_policy(env, v_random, policy)
if optimal:
    print("无更新，最优策略为：")
else:
    print("有更新，更新后策略为：")
print(policy)


# 有更新，更新后策略为：
# [[1. 0. 0. 0.]
#  [0. 0. 0. 1.]
#  [1. 0. 0. 0.]
#  [0. 0. 0. 1.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [0. 0. 0. 1.]
#  [0. 1. 0. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [1. 0. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 1. 0. 0.]
#  [1. 0. 0. 0.]]


########### 策略迭代 ： 策略评估v,q-> 策略更新policy或者 \pi
def iterate_policy(env, gamma=1., torlerant=1e-6):
    policy = np.ones((env.unwrapped.nS, env.unwrapped.nA)) \
             / env.unwrapped.nA  # 初始化为任意一个策略
    while True:
        v = evaluate_policy(env, policy, gamma, torlerant)  # 策略评估  v(s) = \sum_{a} \pi(a|s) * q(s,a)
        if improve_policy(env, v, policy):  # 策略改进  找到使得q(s,a)最大的动作 a' = argmax_a q(s,a),修改[s][a]的值
            break
    return policy, v


# 利用策略迭代求解最优策略
policy_pi, v_pi = iterate_policy(env)
print("状态价值函数 = ")
print(v_pi.reshape(4, 4))
print("最优策略 = ")
print(np.argmax(policy_pi, axis=1).reshape(4, 4))

# SFFF    (S: START,  F: FROZEN, H: HOLE, G: GOAL)
# FHFH
# FFFH
# HFFG
# 测试策略
episode_rewards = [play_policy(env, policy_pi) for _ in range(100)]
print("策略迭代 平均奖励: {}".format(np.mean(episode_rewards)))
# 状态价值函数 =
# [[0.82351246 0.82350689 0.82350303 0.82350106]
#  [0.82351416 0.         0.5294002  0.        ]
#  [0.82351683 0.82352026 0.76469786 0.        ]
#  [0.         0.88234658 0.94117323 0.        ]]
# 最优策略 =    (0：左, 1：下, 2：右, 3：上)
# [[0 3 3 3]   (左, 上，上，上)
#  [0 0 0 0]   (左，左，左，左)
#  [3 1 0 0]   (上，下，左，左)
#  [0 2 1 0]]  (左，右，下，左)
# 策略迭代 平均奖励: 0.74




########有模型价值迭代求解
def iterate_value(env, gamma=1., tolerant=1e-6):
    v = np.zeros(env.observation_space.n)  # 初始化
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            vmax = max(v2q(env,v,s,gamma))   # 更新价值函数 最优状态价值v_*(s) = max_{a \in A} q_*(s,a)
            delta = max(delta, abs(v[s]-vmax))  # 误差值
            v[s] = vmax
        if delta < tolerant:  # 满足迭代需求
            break
    policy = np.zeros((env.observation_space.n, env.action_space.n))  # 计算最优策略
    for s in range(env.observation_space.n):
        a = np.argmax(v2q(env, v, s, gamma))  # 得到最优动作 a' = argmax_a q(s,a)
        policy[s][a] = 1.  # 更新当前状态下的最优动作，即更新策略
    return policy, v

#测试价值迭代算法求解最优策略
policy_vi, v_vi = iterate_value(env)
print("状态价值函数 = ")
print(v_vi.reshape(4,4))
print("最优策略 = ")
print(np.argmax(policy_vi, axis=1).reshape(4,4))
episode_rewards = [play_policy(env, policy_vi) for _ in range(100)]
print("价值迭代 平均奖励:{}".format(np.mean(episode_rewards)))
# 策略迭代和价值迭代得到的最优状态价值函数和最优策略是一样的。
# 状态价值函数 =
# [[0.82351232 0.82350671 0.82350281 0.82350083]
#  [0.82351404 0.         0.52940011 0.        ]
#  [0.82351673 0.82352018 0.76469779 0.        ]
#  [0.         0.88234653 0.94117321 0.        ]]
# 最优策略 =
# [[0 3 3 3]
#  [0 0 0 0]
#  [3 1 0 0]
#  [0 2 1 0]]
# 价值迭代 平均奖励:0.75



### 总结   参考https://zhuanlan.zhihu.com/p/68062905
## 策略迭代
#  1. 计算得到q(s,a)=\sum_{s',r} prob(s',r|s,a) * [r + \gamma v_\pi(s')]
#  2. 策略评估得到v  v(s) = \sum_{a} \pi(a|s) * q(s,a)
#  3. 策略改进\pi <- \pi'  找到使得q(s,a)最大的动作 a' = argmax_a q(s,a),修改[s][a]的值,得到确定性策略
#  4. 以此循环

## 值迭代
# 1. 利用Bellman Optimal Equation 计算得到最优动作价值q_{*}(s,a)=\sum_{s',r} prob(s',r|s,a) * [r + \gamma v_{*}(s')]
# 2. 用最优状态价值 v_*(s) = max_{a \in A} q_*(s,a)
# 3. 策略改进\pi <- \pi'  找到使得q(s,a)最大的动作 a' = argmax_a q(s,a),修改[s][a]的值
# 4. 以此循环