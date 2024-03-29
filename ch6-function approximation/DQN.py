'''
DQN的改进：
    1. 经验回放
    2. 目标网络
'''
'''
源代码存在问题：
    1. target_Q_network 无法进行更新，导致dqn学习的目标一直未能更新。
        具体原因是：在训练初期，每个episode内的step是很难达到100步，导致target_Q_network无法更新，使用的仍然是初始的Q_network的参数。
                 target_Q_network无法更新优化，产生的目标Q值是差的，动作的效果也是差的，而Q_network一直学习的目标Q值就是target_Q_network产生的。
        修改方法：让step_counter成为全局变量，每个episode产生的step都累计，然后到达指定的100step就更新target_Q_network，这个样目标Q值就也在不断优化。
    2. 按照gym环境给的奖励函数，reward值最大只有200，限制了训练效果的进一步提升。


'''


import gym
import numpy as np
import copy
from collections import deque
import torch
from torch import nn
import random
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000



'''定义Q网络类'''
class NN(nn.Module):     #继承与torch的nn.module类
    ## 类构造函数
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()    # 将输入拉直成向量
        # 定义Q网络
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 20),     # 输入层到第1隐藏层的线性部分
            nn.ReLU(),                     # 第1隐藏层激活函数
            nn.Linear(20, 20),             # 第1隐藏层到第2隐藏层的线性部分
            nn.ReLU(),                     # 第2隐藏层激活函数
            nn.Linear(20, output_size)     # 第2隐藏层到输出层
            )

    def forward(self,x):              # 前向传播函数
        x = self.flatten(x)           # 将输入拉直成向量
        logits = self.linear_relu_stack(x)  # 前向传播，预测x的值
        return logits                       # 返回预测值

'''
定义DQN智能体类
'''
class DQN(object):
    def __init__(self, env):
        self.learn_step_counter = 0    # for target updating
        self.replay_size = 0           # for storing memory
        self.replay_buffer = deque()   # 初始化经验回放池

        # 创建预测Q网络实体
        self.Q_network = NN(env.state_dim, env.aspace_size)
        # 创建目标Q网络实体，直接复制预测Q网络
        self.Target_Q_network = copy.deepcopy(self.Q_network)

        # 损失函数
        self.loss_fun = nn.MSELoss(reduction='mean')
        # 随机梯度下降(SGD)优化器
        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), lr=LR)

    ## epsilon-greedy策略函数
    def choose_action(self, state):
        state = torch.from_numpy(np.expand_dims(state,0))  # 转换成张量 在axis=0维度上扩展，(x,y)->(1,x,y)
        state = state.to(torch.float32)  # 张量的默认类型为float32

        # 计算所有动作值
        Q_value = self.Q_network.forward(state)
        # 以epsilon设定动作概率
        A = np.ones(env.aspace_size)*EPSILON/env.aspace_size
        # 选取最大动作值对应的动作
        best = np.argmax(Q_value.detach().numpy())  # Q_value不需要反向传播计算梯度，从当前计算图中分离出来，再转成ndarray
        # 以 1 - epsilon的概率设定贪婪动作
        A[best] += 1-EPSILON
        # 选择动作
        action = np.random.choice(range(env.aspace_size), p=A)
        # 返回动作编号
        return action

    ## 经验回放技术
    def store_transition(self, state, action, reward, next_state, done):
        # 将动作改写成one-hot向量
        one_hot_action = np.eye(env.aspace_size)[action]  # 得到的one-hot结果大小为 (env.aspace_size x action),
                        # env.aspace_size列，action.size行，并且根据action的内容将one-hot向量中对应位置设置为1。
        # 将新数据存入经验回放池
        self.replay_buffer.append((state, one_hot_action,
                                   reward, next_state, done))
        self.replay_size += 1
        # print("replay_buffer_size={}".format(len(self.replay_buffer)))
        # 如果经验回放池溢出，则删除最早经验数据
        if len(self.replay_buffer) > MEMORY_CAPACITY:
            self.replay_buffer.popleft()  # deque是双端队列，可以在队列两端添加和删除，左端为最早的经验数据，因此先删除
        # 经验回放池中数据量多于一个批量batch_size就可以开始训练Q网络
        if len(self.replay_buffer) > BATCH_SIZE:
            self.learn()

    # Q网络训练函数
    def learn(self):
        if (self.learn_step_counter + 1) % TARGET_REPLACE_ITER == 0:  # 目标Q网络参数更新
            self.Target_Q_network.load_state_dict(
                self.Q_network.state_dict())
        self.learn_step_counter += 1
        print("learn_step_counter={}")

        # 从经验回放池中随机抽取一个批量
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
           # minibatch：（state, action, reward, next_state, done）  分别对应  x[0], x[1], x[2], x[3], x[4]
        state_batch = np.array([x[0] for x in minibatch])
        action_batch = np.array([x[1] for x in minibatch])


        # 计算TD目标值
        y_batch = []
        for x in minibatch:   # 对minibatch中每一条MDP数据循环
            if x[4]:          # 如果已经到达终止状态
                y_batch.append(x[2])  # 记录一个mdp的reward
            else:            # 尚未达到终止状态
                next_state = torch.from_numpy(x[3]).unsqueeze(0).to(torch.float32)  # 将x[3]即next_state在行维度上增加一行，并转换为tensor
                q_value_next = self.Target_Q_network(next_state)
                td_target = x[2] + GAMMA * torch.max(q_value_next)
                y_batch.append(td_target.item())  # item()表示从td_target张量中取出该元素值
        y_batch = np.array(y_batch)

        # 将numpy.array数据转换为torch.tensor数据
        state_batch = torch.from_numpy(state_batch).to(torch.float32)     # shape= 32 x 4
        action_batch = torch.from_numpy(action_batch).to(torch.float32)   # shape= 32 x 2
        y_batch = torch.from_numpy(y_batch).to(torch.float32)        # shape= 32 x 1

        # self.Q_network.train()   # 声明训练过程


        # 预测批量值和损失函数
        pred = torch.sum(torch.multiply(self.Q_network(state_batch), action_batch), dim=1)  # dim=1，表示张量只有一个维度，类似于一维数组
        # pred_shape = pred.shape     # shape = 32 x 1
        # print(pred_shape)
        loss = self.loss_fun(pred, y_batch)

        # 误差反向传播，训练Q-网络
        self.optimizer.zero_grad()   # 梯度归零
        loss.backward()              # 求各个参数的梯度值
        self.optimizer.step()        # 误差反向传播修改参数



# 训练函数
def train(env,num_episodes=400, render=False):
    dqn = DQN(env)
    rewards = []  # 每一回合的累积奖励
    # 外层循环指导最大轮次
    for episode in range(num_episodes):
        state = env.reset()  # 环境初始化
        reward_sum = 0  # 当前轮次的累积奖励

        # 内层循环指导最大交互次数或到达终止状态
        while True:
            if render:
                env.render()

            action = dqn.choose_action(state)  # epsilon-贪婪策略选定动作

            next_state, reward, done, _ = env.step(action)  # 交互一个时间步

            # modify the reward
            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            dqn.store_transition(state, action, r, next_state, done)  # 经验回放技术，训练

            reward_sum += r  # 累积折扣奖励
            if dqn.replay_size > MEMORY_CAPACITY:
                dqn.learn()
            # 如果达到终止状态则结束本轮循环
            if done:
                rewards.append(reward_sum)
                break
            state = next_state  # 更新状态
        print("episodes:{} rewards:{}".format(episode, reward_sum))
    # 图示训练过程
    plt.figure('train')
    plt.title('train')
    plt.plot(range(num_episodes), rewards, label='accumulate rewards')
    plt.legend()
    filepath = 'train.png'
    plt.savefig(filepath, dpi=300)
    plt.show()

    # # 测试函数
    # def test(self, num_episodes=100, render=False):
    #     # 循环直到最大测试轮数
    #     rewards = []                    # 每一回合的累积奖励
    #     for episode in range(num_episodes):
    #         state = self.env.reset()    # 环境状态初始化
    #
    #         # 循环直到到达终止状态
    #         reward_sum = 0              # 当前轮次的累积奖励
    #         while True:
    #             if render:
    #                 env.render()
    #             action = self.egreedy_action(state)    # epsilon-贪婪策略选定动作
    #             next_state, reward, end, info = self.env.step(action)   # 交互一个时间步
    #             reward_sum += reward        # 累积奖励
    #             state = next_state          # 状态更新
    #
    #             # 检查是否到达终止状态
    #             if end:
    #                 rewards.append(reward_sum)
    #                 break
    #         print("episodes:{} rewards:{}".format(episode, reward_sum))
    #     score = np.mean(np.array(rewards))  # 计算测试得分
    #
    #     # 图示测试结果
    #     plt.figure('test')
    #     plt.title('test: score=' + str(score))
    #     plt.plot(range(num_episodes), rewards, label='accumulate rewards')
    #     plt.legend()
    #     filepath = 'test.png'
    #     plt.savefig(filepath, dpi=300)
    #     plt.show()
    #
    #     return score  # 返回测试得分




'''主程序'''
if __name__ == '__main__':
    # 加载环境
    env = gym.make('CartPole-v0')   # 导入CartPole环境
    env.state_dim = env.observation_space.shape[0]  # 获取行维度
    env.aspace_size = env.action_space.n            # 离散动作个数

    train(env, render=True)                # 训练智能体
    # agent.test(render=True)                 # 测试智能体


