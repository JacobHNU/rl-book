import gym

env = gym.make("MountainCar-v0")
print('观测空间={}'.format(env.observation_space))
print('动作空间={}'.format(env.action_space))
print('观测范围={}~{}'.format(env.observation_space.low, env.observation_space.high))
print('动作数={}'.format(env.action_space.n))


# output
# 观测空间=Box([-1.2  -0.07], [0.6  0.07], (2,), float32)
# 动作空间=Discrete(3)
# 观测范围=[-1.2  -0.07]~[0.6  0.07]      位置x范围[-1.2, 0.6]   速度v范围[0,0.07]
# 动作数=3   向左加速、不加速、向右加速

class BespkikeAgent:
    def __init__(self, env):
        pass

    def decide(self, observation):  # 决策
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action  # 返回动作

    def learn(self, *args):  # 学习
        pass

def play_montecarlo(env, agent, render=False, train=False):
    episode_reward = 0.  # 记录回合总奖励， 初始化为0
    observation= env.reset()  # 重置游戏环境, 开始新回合
    while True:  # 不断循环
        if render:  # 是否显示
            env.render()  # 显示图形界面,图形界面可以用env.close()语句关闭
        action = agent.decide(observation)
        next_observation, reward, terminated, _ = env.step(action)  # 执行动作
        episode_reward += reward  # 收集回合奖励
        if train:  # 判断是否训练智能体
            agent.learn(observation, action, reward, terminated, _)  # 学习
        if terminated:  # 回合结束，跳出循环
            break
        observation = next_observation
    return episode_reward

agent = BespkikeAgent(env)
total_reward = 0.
for _ in range(10):
    episode_reward = play_montecarlo(env, agent, render=True)
    print('回合奖励={}'.format(episode_reward))
    total_reward += episode_reward
print('平均奖励={}'.format(total_reward/10))
env.close()
