# 2023/02/16
import numpy as np
np.random.seed(0)
import pandas as pd
import gym
# from IPython.display import display
space_names =['观测空间','动作空间','奖励范围','最大步数']
df = pd.DataFrame(columns=space_names)

for env in gym.envs.registry.all():
    env_id =env.id
    # print(env_id)
    try:
        env = gym.make(env_id)
        observation_space = env.observation_space
        action_space = env.action_space
        reward_range = env.reward_range
        max_episode_steps = None
        if isinstance(env, gym.wrappers.time_limit.TimeLimit):
            max_episode_steps = env._max_episode_steps
        df.loc[env_id] = [observation_space, action_space,reward_range, max_episode_steps]
    except:
        pass
with pd.option_context('display.max_rows', None):
     # display(df)
     print(df)

