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

import numpy as np
np.random.seed(0)
import plot_func as plot
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
#

## 同策回合更新
# 回合更新预测
def ob2state(observation):
    return observation[0], observation[1], int(observation[2])

def evaluate_action_monte_carlo(env, policy, episode_num=500000):
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        # 玩一个回合
        state_action = []
        observation = env.reset()
        print('observation={}'.format(observation))
        while True:
            state = ob2state(observation)
            print('state={}'.format(state))
            # print('policy[state]={}'.format(policy[state]))
            # print('env.action_space.n={}'.format(env.action_space.n))
            action = np.random.choice(env.unwrapped.action_space.n, p=policy[state])  # action_space.n需要与len(p)的大小对应，就能确定每个action对应的选取概率
            print('action={}'.format(action))
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
# 庄家可见点数下的二维数组(2,2)，第一维表示是否将A置为11，第二维表示动作分别都是一个size=2的一维数组(不要牌，要牌)置1有效。
## 举个例子： (19,2,1,0)  闲家牌点数19（统一从0开始定位到第19个三维数组），庄家可见牌点数2（第19个三维数组内的第2个二维数组），
## A设置为11为True（即为1）（第19个三维数组内的第2个二维数组内的第1行），玩家不要牌0（第19个三维数组内的第2个二维数组内的第1行的第0个位置置1）
policy[20:, :, :, 0] = 1  # >=20 时不再要牌
# （第一个维度一共22个三维数组(11,2,2)的）20,21个三维数组的最后一个维度（size=2的一个一维数组）的数组的第0个位置上置为1
policy[:20, :, :, 1] = 1  # < 20 时继续要牌
#print('policy.={}'.format(policy))
# 第0到19个三维数组的最后一个维度数组（size=2）的第1个位置上置为1
q = evaluate_action_monte_carlo(env, policy)  # 动作价值
v = (q * policy).sum(axis=1)  # 状态价值


def play_once(env):
    total_reward = 0
    observation, _ = env.reset()
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

print("随机策略 奖励：{}".format(play_once(env)))
plot(v)