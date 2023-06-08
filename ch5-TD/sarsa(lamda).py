'''
涉及到资格迹，对资格迹不太了解，涉及到信度分配。简要介绍一下sarsa(λ)和sarsa的区别：
  sarsa是每次获取到reward之后只更新到reward的前一步，而sarsa(λ)是更新获取到reward的前λ步，
  即是sarsa在没有获得reward之前，当前步的Q值其实是没有任何变化的，直到获得reward之后才会更新前一步；
  而sarsa(λ)则会对获得reward的所有步都进行更新，离reward越近的步越重要，越远的步则越不重要（由λ控制衰减幅度）。
  λ是在[0,1]之间取值，如果λ=0，sarsa(λ)就是sarsa，只更新获取到reward前一步
  λ=1，sarsa(λ)就是回合更新，更新的是获取到reward前所有经历过的步。
'''

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt



class SARSALamdaAgent:
    def __init__(self, env, lambd=0.6, beta=1.,
                 gamma=0.9, learning_rate=0.1, epsilon=.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.beta = beta
        self.epsilon = epsilon
        self.action_n = env.aspace_size
        self.nA = env.aspace_size
        self.q = defaultdict(lambda: np.zeros(self.nA))
        self.e = defaultdict(lambda: np.zeros(self.nA))

    def decide(self, state):
        if np.random.uniform() > self.epsilon:  # np.random.uniform()均匀分布随机采样
            # self.epsilon=0.1,  >0.1 则表示选取的是用动作价值(q0)确定的策略决定动作A选取的概率大于其余的随机选择的概率
            action = np.argmax(self.q[state])
        else:
            # 其余就是0.1，则表示0.1的
            action = np.random.randint(self.action_n)  # np.random.randint() 返回一个随机整型数
        return action

    def learn(self, state, action, reward, next_state, done, next_action):
        # 更新资格迹
        self.e[state][action] = 1. + self.beta*self.e[state][action]

        # 更新价值
        u = reward + self.gamma * \
            self.q[next_state][next_action] * (1. - done)
        td_error = u - self.q[state][action]

        for s in env.get_sspace():
            for a in env.get_aspace():
                self.e[s][a] = self.lambd * env.gamma * self.e[s][a]        # 资格迹衰减
                self.q[s][a] += self.learning_rate * self.e[s][a] * td_error  # 策略评估


def play_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, done ,_ = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation) # 终止状态时此步无意义
        if train:
            agent.learn(observation, action, reward, next_observation,
                    done, next_action)
        if done:
            break
        observation, action = next_observation, next_action
    return episode_reward


import WindyWorld
env = WindyWorld.WindyWorldEnv()
agent = SARSALamdaAgent(env)

# 训练
episodes = 1000
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_sarsa(env, agent, train=True)
    episode_rewards.append(episode_reward)

plt.plot(episode_rewards)
plt.show()

# 测试
agent.epsilon = 0. # 取消探索

episode_rewards = [play_sarsa(env, agent, train=False) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))