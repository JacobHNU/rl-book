'''
   利用动作价值函数Q和状态价值函数V之差，得到优势函数A
   然后为了使得Q=>V+A, V+A=>Q，采用优势函数A的导出量作为A_等效，即Q=V+A_等效,等式具有唯一值

   在原DQN的基础上，
   1. 分解原来的神经网络为公共部分，优势函数部分和状态价值网络部分；
   2. forward前向传播时，计算Q=V+A_等效  = V + A - A.mean()
   3. 在Q网络训练时，
       3.1 利用当前Q网络得到的动作价值Q获取到最优动作a=argmax(Q),
       3.2 利用目标Q网络得到的动作价值Q’
       3.3 Q'_max = torch.dot(Q', a)获得目标Q网络的贪婪动作值
       3.4 计算td_target时利用Q'_max, td_target = reward + gamma * Q’_max

'''

import gym
import numpy as np
import copy
from collections import deque
import torch
from torch import nn
import random
import matplotlib.pyplot as plt


'''定义Q网络类'''
class NN(nn.Module):     #继承与torch的nn.module类
    ## 类构造函数
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.flatten = nn.Flatten()    # 将输入拉直成向量

        # 公共网络
        self.public_stack = nn.Sequential(  #
            nn.Linear(input_size, 20),     # 输入层到第1隐藏层的线性部分
            nn.ReLU(),                     # 第1隐藏层激活函数
            nn.Linear(20, 20),             # 第1隐藏层到第2隐藏层的线性部分
            nn.ReLU(),                     # 第2隐藏层激活函数
            nn.Linear(20, 20),              # 第2隐藏层到输出层
            nn.ReLU()
            )
        # 优势网络部分
        self.advantage_stack = nn.Sequential(
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,output_size)
            )
        # 状态价值网络
        self.stateValue_stack = nn.Sequential(
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20, 1)
            )

    def forward(self,x):              # 前向传播函数
        x = self.flatten(x)           # 将输入拉直成向量
        pub = self.public_stack(x)
        adv = self.advantage_stack(pub)
        staVal=self.stateValue_stack(pub)
        Q = staVal+adv-adv.mean(1).unsqueeze(1).expand(
                                x.size(0), self.output_size)  # 前向传播，预测x的值
        return Q                     # 返回预测值

'''
定义DQN智能体类
'''
class DuelingDQN():
    def __init__(self, env, epsilon=0.1, learning_rate=0.01,
                 replay_size=2000, batch_size=32, target_replace_iter=100):
        self.replay_buffer = deque()   # 初始化经验回放池
        self.env = env                 # 环境模型
        self.epsilon = epsilon         # epsilon-greedy策略的参数
        self.learning_rate = learning_rate  # 学习率
        self.replay_size = replay_size      # 经验回放池最大容量
        self.batch_size = batch_size        # 批量尺度
        self.target_replace_iter = target_replace_iter
        self.learn_step_counter = 0

        self.create_Q_network()          # 生成Q网络实体
        self.create_training_method()    # Q网络优化器

    ## Q网络生成函数
    def create_Q_network(self):
        # 创建预测Q网络实体
        self.Q_network = NN(self.env.state_dim,
                            self.env.aspace_size)
        # 创建目标Q网络实体，直接复制预测Q网络
        self.Q_network_t = copy.deepcopy(self.Q_network)

    ## Q网络优化器生成函数
    def create_training_method(self):
        # 损失函数
        self.loss_fun = nn.MSELoss(reduction='mean')
        # 随机梯度下降(SGD)优化器
        self.optimizer = torch.optim.Adam(self.Q_network.parameters(),
                                         lr=self.learning_rate)

    ## epsilon-greedy策略函数
    def egreedy_action(self, state):
        state = torch.from_numpy(np.expand_dims(state,0))  # 转换成张量 在axis=0维度上扩展，(x,y)->(1,x,y)
        state = state.to(torch.float32)  # 张量的默认类型为float32

        # 计算所有动作值
        Q_value = self.Q_network.forward(state)
        # 以epsilon设定动作概率
        A = np.ones(self.env.aspace_size)*self.epsilon/self.env.aspace_size
        # 选取最大动作值对应的动作
        best = np.argmax(Q_value.detach().numpy())  # Q_value不需要反向传播计算梯度，从当前计算图中分离出来，再转成ndarray
        # 以 1 - epsilon的概率设定贪婪动作
        A[best] += 1-self.epsilon
        # 选择动作
        action = np.random.choice(range(self.env.aspace_size), p=A)
        # 返回动作编号
        return action

    ## 经验回放技术
    def perceive(self, state, action, reward, next_state, done):
        # 将动作改写成one-hot向量
        one_hot_action = np.eye(self.env.aspace_size)[action]  # 得到的one-hot结果大小为 (env.aspace_size x action),
                        # env.aspace_size列，action.size行，并且根据action的内容将one-hot向量中对应位置设置为1。
        # 将新数据存入经验回放池
        self.replay_buffer.append((state, one_hot_action,
                                   reward, next_state, done))
        # 如果经验回放池溢出，则删除最早经验数据
        if len(self.replay_buffer) > self.replay_size:
            self.replay_buffer.popleft()  # deque是双端队列，可以在队列两端添加和删除，左端为最早的经验数据，因此先删除
        # 经验回放池中数据量多于一个批量batch_size就可以开始训练Q网络
        if len(self.replay_buffer) > self.batch_size:
            self.train_Q_network()

    # Q网络训练函数
    def train_Q_network(self):
        # 从经验回放池中随机抽取一个批量
        minibatch = random.sample(self.replay_buffer, self.batch_size)
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
                # 用当前Q-网络计算下一状态的动作值
                value = self.Q_network.forward(next_state)
                # 求最大动作值对应的动作
                action_max = torch.argmax(value).item()
                action_max_onehot = torch.eye(self.env.aspace_size)[action_max]
                # 用目标Q网络计算动作值
                value_next = self.Q_network_t(next_state).squeeze(0)
                # 贪婪动作的动作值
                value_next_max = torch.dot(value_next,action_max_onehot)
                # TD目标值
                td_target = x[2]+self.env.gamma * value_next_max
                y_batch.append(td_target.item())  # item()表示从td_target张量中取出该元素值
        y_batch = np.array(y_batch)

        # 将numpy.array数据转换为torch.tensor数据
        state_batch = torch.from_numpy(state_batch).to(torch.float32)     # shape= 32 x 4
        action_batch = torch.from_numpy(action_batch).to(torch.float32)   # shape= 32 x 2
        y_batch = torch.from_numpy(y_batch).to(torch.float32)        # shape= 32 x 1

        self.Q_network.train()   # 声明训练过程
        # output = self.Q_network(state_batch)
        # output_shape = output.shape
        # action_batch_shape = action_batch.shape
        # print("Q_network output={}".format(output_shape))

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
    def train(self, num_episodes=400, num_steps=2000, render=False):
        # 外层循环指导最大轮次
        rewards = []                      # 每一回合的累积奖励
        for episode in range(num_episodes):
            state = self.env.reset()     # 环境初始化

            # 内层循环指导最大交互次数或到达终止状态
            reward_sum = 0               # 当前轮次的累积奖励
            for step in range(num_steps):
                env.render()
                self.learn_step_counter += 1
                action = self.egreedy_action(state)  # epsilon-贪婪策略选定动作
                next_state, reward, done, _ = self.env.step(action)  # 交互一个时间步

                # modify the reward  修改奖励确实能加速训练效果的提升
                # x, x_dot, theta, theta_dot = next_state
                # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                # r = r1 + r2

                reward_sum += reward     # 累积折扣奖励

                self.perceive(state, action, reward, next_state, done)  # 经验回放技术，训练
                state = next_state       # 更新状态
                if (self.learn_step_counter + 1) % self.target_replace_iter == 0:  # 目标Q网络参数更新
                    self.Q_network_t.load_state_dict(
                        self.Q_network.state_dict())
                # 如果达到终止状态则结束本轮循环
                if done:
                    rewards.append(reward_sum)
                    break
            print("episodes:{} rewards:{}".format(episode, reward_sum))
        # 图示训练过程
        plt.figure('train')
        plt.title('train')
        plt.plot(range(num_episodes), rewards, label='accumulate rewards')
        plt.legend()
        filepath = 'train2015.png'
        plt.savefig(filepath, dpi=300)
        plt.show()

    # 测试函数
    def test(self, num_episodes=100, render=False):
        # 循环直到最大测试轮数
        rewards = []                    # 每一回合的累积奖励
        for episode in range(num_episodes):
            state = self.env.reset()    # 环境状态初始化

            # 循环直到到达终止状态
            reward_sum = 0              # 当前轮次的累积奖励
            while True:
                if render:
                    env.render()
                action = self.egreedy_action(state)    # epsilon-贪婪策略选定动作
                next_state, reward, end, info = self.env.step(action)   # 交互一个时间步
                reward_sum += reward        # 累积奖励
                state = next_state          # 状态更新

                # 检查是否到达终止状态
                if end:
                    rewards.append(reward_sum)
                    break
            print("episodes:{} rewards:{}".format(episode, reward_sum))
        score = np.mean(np.array(rewards))  # 计算测试得分

        # 图示测试结果
        plt.figure('test')
        plt.title('test: score=' + str(score))
        plt.plot(range(num_episodes), rewards, label='accumulate rewards')
        plt.legend()
        filepath = 'test2015.png'
        plt.savefig(filepath, dpi=300)
        plt.show()

        return score  # 返回测试得分

'''主程序'''
if __name__ == '__main__':
    # 加载环境
    env = gym.make('CartPole-v0')   # 导入CartPole环境
    env.gamma = 0.9                   # 折扣系数
    env.state_dim = env.observation_space.shape[0]  # 获取行维度
    env.aspace_size = env.action_space.n            # 离散动作个数

    agent = DuelingDQN(env)                 # 创建一个DuelingDQN智能体
    agent.train(render=True)                # 训练智能体
    agent.test(render=True)                 # 测试智能体

