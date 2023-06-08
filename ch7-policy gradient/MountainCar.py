'''
一般认为，如果智能体在连续100个回合中的平均步数 小于等于 110, 认为问题解决
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07

'''



import math
import numpy as np
from gym import spaces
from gym.utils import seeding


class MountainCarEnv():
    def __init__(self):
        self.min_position = -1.2  # 最低点
        self.max_position = 0.6  # 最高点
        self.max_speed = 0.07  # 最大速度
        self.goal_position = -0.2  # 目标高度
        self.goal_velocity = 0  # 目标速度
        self.force = 0.001  # 推力
        self.gravity = 0.0025  # 重力
        self.time = None  # 一个回合持续时间步
        self.viewer = None

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def step(self, action):
        position, velocity = self.state
        # 交互过后的速度
        velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity,-self.max_speed, self.max_speed)  # clip截取函数，限定在最大最小值之间
        # 交互过后的速度
        position += velocity
        position = np.clip(position,self.min_position, self.max_position)  # 时间按照交互一次增加1个时间步
        # 考虑小车特殊位置
        if (position == self.min_position and velocity<0):
            velocity=0
        self.state = [position, velocity]
        self.time += 1

        # 计算goal_position和goal_velocity时的奖励，
        if(position >= self.goal_position and velocity>=self.goal_velocity):
            done = True
            reward = 0
            info = 'Goal Obtained'
        elif self.time > 150:
            done = True
            reward = -1
            info = 'Maxium Timesteps'
        else:
            done = False
            reward = -1
            info = 'Goal Obtained'    # ?

        return self.state, reward, done, info

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")


    def reset(self):
        self.state = [self.np_random.uniform(low=-0.6, high=-0.4), 0]
        self.time = 0

        return self.state


if __name__ == '__main__':
    env = MountainCarEnv()
    for i in range(400):
        s = env.reset()
        rewards = []
        while True:
            env.render()
            prob = np.random.rand(3, )
            prob = prob / np.sum(prob)
            a = np.random.choice(np.arange(3), p=prob)
            s_, r, done, info = env.step(a)
            rewards.append(r)
            print(i, s, a, s_, done, info)

            if done:
                break







