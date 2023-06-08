import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class DoubleQLearningAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=.1):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.aspace_size
        self.q0 = defaultdict(lambda: np.zeros(self.action_n))
        self.q1 = defaultdict(lambda: np.zeros(self.action_n))

    def decide(self, state):
        if np.random.uniform() > self.epsilon:  # np.random.uniform()均匀分布随机采样
            # self.epsilon=0.1,  >0.1 则表示选取的是用动作价值(q0 + q1) = q(s,.)确定的策略决定动作A选取的概率大于其余的随机选择的概率
            action = np.argmax(self.q0[state] + self.q1[state])
        else:
            # 其余就是0.1，则表示0.1的
            action = np.random.randint(self.action_n)  # np.random.randint() 返回一个随机整型数
        return action

    def learn(self, state, action, reward, next_state, done):
        if np.random.randint(2):
            self.q0, self.q1 = self.q1, self.q0
        a = np.argmax(self.q0[next_state])
        u = reward + self.gamma * self.q1[next_state][a] * (1. - done)
        td_error = u - self.q0[state][action]
        self.q0[state][action] += self.learning_rate * td_error


def play_qlearning(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, next_observation, done)
        if done:
            break
        observation = next_observation
    return episode_reward


import WindyWorld

env = WindyWorld.WindyWorldEnv()

lr_list = [0.1, 0.3]

# 训练
episodes = 9000
episodes_reward_rec = [[] for i in range(len(lr_list))]

for lr_idx, lr in enumerate(lr_list):
    agent = DoubleQLearningAgent(env, learning_rate=lr)
    for episode in range(episodes):
        episode_reward = play_qlearning(env, agent, train=True)
        episodes_reward_rec[lr_idx].append(episode_reward)

# 用表格表示最终策略
P_table = np.ones((env.world_height, env.world_width)) * np.inf
for state in env.get_sspace():
    P_table[state[0]][state[1]] = np.argmax(agent.q0[state])

for idx, ep_reward in enumerate(episodes_reward_rec):
    plt.plot(range(len(ep_reward)), ep_reward, label='lr={}'.format(lr_list[idx]))

plt.xlabel("Episodes")
plt.ylabel("Episodes_rewards")
plt.legend()
plt.show()

# 测试
agent.epsilon = 0.  # 取消探索
episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
print('平均回合奖励={}/{}={}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))
print('P_table =', P_table)
for state in env.get_sspace():
    print(state, agent.q0[state])
