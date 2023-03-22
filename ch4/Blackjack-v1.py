#### ch3复习
# 数学底层原理是Banach不动点定理，可以用迭代的方法求解Bellman期望方程
#
# 策略迭代 利用动态规划更新策略
# 利用Bellman期望公式，通过bootstrap方式得到状态价值V(s)
# 构造q(s,a), 用状态价值V表示动作价值q(s,a) ，确定性策略
# 更新策略policy, \pi(s) = \argmax q(s,a)

# 值迭代
# 利用Bellman最优公式， 算出最优动作价值函数q_*(s,a)
# 用最优动作价值q_*(s,a)表示最优状态价值 v_*(s,a)=\max_{a \in A} q_*(s,a)
# 再将最优状态价值V_*(s)表示最优动作价值q_*(s,a)
# 最后更新策略 policy, \pi(s)=\argmax q_*(s,a)

#### Model-Free 无模型环境
# 无模型的机器学习算法在没有数学描述的环境（为环境建立精确的数学模型及其困难）下，
# 只能依靠经验（如轨迹的样本）学习出给定策略的价值函数和最优策略
#### 回合更新价值迭代
## 回合更新策略评估：用Monte Carlo方法来估计这个期望
## 在model-based下，状态价值和动作价值可以互相表示
# 任意策略的价值函数满足Bellman期望方程，借助动力p（环境转移模型），可以用状态价值函数表示动作价值函数
# 状态价值函数->动作价值函数  q_\pi(s,a) = \sum_{s'} p(s'|s,a)*v_\pi(s')
# 借助策略\pi的表达式，可以动作价值表示状态价值
# 动作价值->状态价值 v_\pi(s) = \sum_{a} \pi(a|s)q_\pi(s,a)
## 在model-free 下，p的表达式未知，只能用动作价值表示状态价值，反之则不行。
# 由于策略改进可以仅由动作价值函数确定，因此学习问题中，动作函数往往更加重要。（\pi(s) = \argmax_{a \in A} q_\pi(s,a)）
# 目标是Q(s,a)
# 实际求解的是样本均值：G(s,a)，然后用样本均值去近似目标

# 首次访问回合更新——每个回合只采用第一次访问的回报样本更新价值函数
# 每次访问回合更新——采用回合内全部的回报样本值更新价值函数

import numpy as np
np.random.seed(0)
import plot_func as plt
import gym

env = gym.make("Blackjack-v1")
print('观察空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('动作数量 = {}'.format(env.action_space.n))
# 观察空间 = Tuple(Discrete(32), Discrete(11), Discrete(2))  （玩家的点数和，庄家可见牌的点数和，是否将A牌算为11）
# 动作空间 = Discrete(2)   （0 表示玩家不再要更多的牌， 1 表示玩家再要一张牌）stick (0), and hit (1).
# 动作数量 = 2

#  Rewards
#    - win game: +1
#    - lose game: -1
#    - draw game: 0
#    - win game with natural blackjack:

## 庄家策略
# 小于17点，继续要牌；>=17点，不要牌。
# 玩家策略

## 同策回合更新
# 回合更新预测
def ob2state(observation):
    return observation[0], observation[1], int(observation[2])

def evaluate_action_monte_carlo(env, policy, episode_num=10000):
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        # 玩一个回合
        state_action = []
        observation = env.reset()
        # print('observation={}'.format(observation))
        while True:
            state = ob2state(observation)
            # print('state={}'.format(state))
            # print('policy[state]={}'.format(policy[state]))
            # print('env.action_space.n={}'.format(env.action_space.n))
            action = np.random.choice(env.unwrapped.action_space.n, p=policy[state])  # action_space.n需要与len(p)的大小对应，就能确定每个action对应的选取概率
            # print('action={}'.format(action))
            state_action.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break  # 回合结束
        g = reward  # 回报
        for state, action in state_action:
            c[state][action] += 1.   # 增量法
            q[state][action] += (g - q[state][action]) / c[state][action]  # 更新动作价值
    return q

policy = np.zeros((22,11,2,2))
# size=(22,11,2,2) 第一个维度（0~21）表示闲家牌的点数，也就是在0~21的牌的点数下的三维数组(11,2,2)，0~10表示庄家的可见牌点数和，
# 庄家可见点数下的二维数组(2,2)，第一维表示是否将A置为1，第二维表示动作分别都是一个size=2的一维数组(不要牌，要牌)置1有效。
## 举个例子： (19,2,1,0)  闲家牌点数19（统一从0开始定位到第19个三维数组），庄家可见牌点数2（第19个三维数组内的第2个二维数组），
## A设置为11为True（即为1）（第19个三维数组内的第2个二维数组内的第1行），玩家不要牌0（第19个三维数组内的第2个二维数组内的第1行的第0个位置置1）
policy[20:, :, :, 0] = 1  # >=20 时不再要牌
# （第一个维度一共22(从0点到21点)个三维数组(11,2,2)的）20,21个三维数组的最后一个维度（size=2的一个一维数组）的数组的第0个位置上置为1
policy[:20, :, :, 1] = 1  # < 20 时继续要牌
# print('policy.={}'.format(policy))
# 第0到19个三维数组的最后一个维度数组（size=2）的第1个位置上置为1
# q = evaluate_action_monte_carlo(env, policy)  # 动作价值
# v = (q * policy).sum(axis=1)  # 状态价值



def play_once(env):
    total_reward = 0
    observation = env.reset()
    print('观测 = {}'.format(observation))
    while True:
        print('玩家 = {}, 庄家 = {}'.format(env.player, env.dealer))
        action = np.random.choice(env.action_space.n)
        print('动作 = {}'.format(action))
        observation, reward, done, _ = env.step(action)
        print('观测 = {}, 奖励 = {}, 结束指示 = {}'.format(
                observation, reward, done))
        total_reward += reward
        if done:
            return total_reward  # 回合结束

# print("随机策略 奖励：{}".format(play_once(env)))
# plt.plot(v)
##
# 观测 = (15, 10, False), 奖励 = 0.0, 结束指示 = False
# 玩家 = [3, 10, 2], 庄家 = [10, 2]
# 动作 = 1
# 观测 = (16, 10, False), 奖励 = 0.0, 结束指示 = False
# 玩家 = [3, 10, 2, 1], 庄家 = [10, 2]
# 动作 = 0
# 观测 = (16, 10, False), 奖励 = -1.0, 结束指示 = True
#随机策略 奖励：-1.0

## 同策最优策略求解
 # 对于同策回合更新评估算法，每次迭代会更新价值估计。如果在更新价值估计后，进行策略改进，那么就会得到新的策略，不断更新，就有希望找到最优策略。
 # 若回合更新总是从s(开始)出发，策略初始化为确定性策略（\pi(s(开始)) = \pi(s(中间))）=a(去终止)，则很难找到最优策略\pi_*(s(开始))=a(去中间)
 # 因此提出起始探索，让所有可能的状态动作对都成为可能的回合起点。这样就不会遗漏任何状态动作对。  但理论上不能保证同策回合更新算法是否总能收敛到最优策略
# 带起始探索的回合更新

def monte_carlo_with_exploring_start(env, episode_num=10000):
    policy = np.zeros((22,11,2,2))
    policy[:,:,:,1] = 1.
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        # 随机选择其实状态和起始动作
        state = (np.random.randint(12,22),  # 一般只关注12到21点的区间，小于十二点则必然会选择继续要牌的动作
                 np.random.randint(1,11),   # 关注庄家亮牌的点数1到10
                 np.random.randint(2))      # 是否将A置为11
        action = np.random.randint(2)       # 动作选择：要牌或者不要牌
        # 玩一回合
        env.reset()
        if state[2]:  # 有A   state[闲家点数，庄家亮牌点数，A是否置为11]
            env.player = [1, state[0]-11]
        else:   # 没有 A
            if state[0] == 21:
                env.player = [10, 9, 2]
            else:
                env.player = [10, state[0] - 10]
        env.dealer[0]=state[1]
        # print("env.player={}".format(env.player))
        # print("env.dealer={}".format(env.dealer))
        state_actions = []
        while True:
            state_actions.append((state,action))
            observation, reward, done, _ = env.step(action)
            if done:
                break  #回合结束
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
        g = reward  # 回报
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
            a = q[state].argmax()
            policy[state] = 0.
            policy[state][a] = 1.
    return policy, q

policy, q = monte_carlo_with_exploring_start(env)
v = q.max(axis=-1)
print("optimal policy ={}".format(policy.argmax(-1)))
print("v={}".format(v))

plt.plot(policy.argmax(-1))
plt.plot(v)

