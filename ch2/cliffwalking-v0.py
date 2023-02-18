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
# 动作空间=Discrete(4)   上，下，左，右
# 状态数量=48, 动作数量=4
# 地图大小=(4, 12)

def play_once(env, policy):
    total_reward=0
    state = env.reset()
    while True:
        loc = np.unravel_index(state, env.shape)
        print('状态={}, 位置={}'.format(state, loc), end=' ')
        action = np.random.choice(env.nA, p=policy[state])
        next_state, reward, done, _= env.step(action)
        print('动作={}, 奖励={}'.format(action, reward))
        total_reward += reward
        if done:
            break
        state = next_state
    return total_reward

actions = np.ones(env.shape, dtype=int)
actions[-1,:]=0
actions[:, -1] = 2
optimal_policy = np.eye(4)[actions.reshape(-1)]

total_reward = play_once(env, optimal_policy)
print('回合奖励={}'.format(total_reward))