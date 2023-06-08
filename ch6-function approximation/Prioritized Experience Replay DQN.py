'''
优先级的定义应该满足两个条件：
（1） 优先级在数值上应该和误差绝对值成单调递增关系，这是为了满足误差绝对值较大（即优先级较大）的样本获得更大的被抽样的机会；
（2） 优先级数值应大于0，这是为了保证每一个样本都有机会被抽样，即抽样概率大于0.
基于比例的样本优先级
    按照误差的占比  err = U - q(s,q;w)  , p_i = err_i / sum(p_k)      其中p_k = err_k
基于排序优先的样本优先级
    p_i = 1/rank(i)   i为第 i 个样本的从大到小排序的位置

随机优先级采样
1. 采样方法
（1） 贪婪优先级采样  —— p_i =p_i/sum(p_k) 完全按照优先级排序采样
（2） 一致随机采样   —— p_i = 1 / sum(1)  均匀采样
（3） 随机优先级采样
2. 样本被采样的基本原则
（1） 样本被采样的概率应该和样本优先级成正相关关系；
（2） 每一个样本都应该有机会被采样，即被采样的概率大于0.
3. Sum-Tree 随机优先级采样
（1） 二叉树，最底层叶节点存储每个样本的优先级，每个树枝节点的值等于两个分叉的和，顶端节点就是所有叶节点优先级的和
（2） 满二叉树的性质
    叶子结点的个数：leaf_num,
    所有结点的个数：total_num,
    则有：total_num = leaf_num*2-1 # 已知叶子结点数目可以推算出总的结点数目

    父亲结点下标（序号）：p_idx
    左侧儿子结点和右侧儿子结点下标：L_idx, R_idx
    则有：p_idx = (L_idx-1)//2 = (R_idx-1)//2  # //:整除  已知任意一个子结点可以找到其父亲结点
     L_idx = p_idx*2+1, R_idx = L_idx+1  # 已知父亲结点可以找到两个子结点

    父亲结点的数值等于两个子结点数值的和。

    https://zhuanlan.zhihu.com/p/165134346?utm_source=wechat_session&utm_id=0

    存储数据：
    1. 只在叶子结点存储优先级信息，存储时从左向右逐个存取，更新对应的父结点，data_pointer的取值范围[0,data_num)
    data_point  # transition 中的数据指针 0-data_num
    caplitity     # 叶子结点数
    caplitity-1   # 最左侧叶子结点的下标
    idx = caplitity - 1 + data_point # 当前数据的优先级就存储在idx指向的叶子结点
    transition[data_point] = data   # 将当前数据存储到transition对应的位置
'''
from random import random

import gym
import numpy as np
import copy
import torch
from torch import nn
import matplotlib.pyplot as plt

'''
定义Sum-Tree类
'''


class SumTree():
    def __init__(self, bufferSize):
        self.bufferSize = bufferSize  # SumTree 叶节点数量 = 经验回放池容量
        self.tree = np.zeros(2 * bufferSize - 1)  # 储存SumTree的所有节点数值
        self.Transition = np.zeros(bufferSize, dtype=object)  # 储存经验数据，对应多有的叶节点
        self.tranDataIndex = 0  # 经验数据的索引

    ## 向SumTree中增加一个数据
    def add(self, priority, expData):
        '''
        向sumTree中添加一个经验数据，首先将经验数据存入到 tranDataIndex 位置，
        然后标记经验数据在sumTree中的位置sumTreeIndex，
        将tranDataIndex位置经验数据的优先级存入到对应的sumTreeIndex位置，更新sumTree
        :param priority:  优先级
        :param expData: 经验数据
        :return:
        '''
        # tranDataIndex序号对应的经验数据在sumTree中的位置为sumTreeIndex
        sumTreeIndex = self.tranDataIndex + self.bufferSize - 1  # sumTreeIndex标记经验数据在树中的位置
        self.Transition[self.tranDataIndex] = expData  # 将expData存入tranDataIndex位置

        self.update(sumTreeIndex, priority)  # 将经验数据的优先级存入到树中对应的位置，并更新sumTree
        self.tranDataIndex += 1  # 添加了一个经验数据，经验数据的指针加一
        if self.tranDataIndex >= self.bufferSize:
            self.tranDataIndex = 0  # 若容量已满，将叶节点指针拨回0

    ## 在sumTreeIndex位置添加priority，更新sumTree
    def update(self, sumTreeIndex, priority):
        '''
        将经验数据对应的优先级存入到sumTree中对应的位置，
        然后向上回溯，更新父节点的优先级，
        直到根节点。
        :param sumTreeIndex:  经验数据对应的优先级的位置
        :param priority:  优先级
        :return:
        '''
        change = priority - self.tree[sumTreeIndex]  # sumTreeIndex位置优先级的改变量
        self.tree[sumTreeIndex] = priority  # 将优先级存入经验数据所在位置的叶节点
        while sumTreeIndex != 0:  # 回溯至根节点
            sumTreeIndex = (sumTreeIndex - 1) // 2  # 父节点， //：先做除法，在向下取整
            self.tree[sumTreeIndex] += change

    ## 根据value抽样
    def getLeaf(self, value):
        '''
        根据value遍历sumTree，小于左子节点则遍历左子树；大于左子节点，减去左子节点数值后，遍历右子树
        :param value:
        :return: leafIndex, self.tree[leafIndex], self.Transition[tranDataIndex]
                返回value所在的叶子节点，叶子节点的优先级，和叶子节点位置的经验数据
        '''
        parentIdx = 0  # 父节点索引
        while True:
            lChildren = 2 * parentIdx + 1  # 左子节点索引
            rChileren = lChildren + 1  # 右子节点索引
            if lChildren >= self.bufferSize:  # 遍历到底了
                leafIndex = parentIdx  # 父节点变成叶节点
                break
            else:
                if value <= self.tree[lChildren]:  # value小于左子节点数值，遍历左子树
                    parentIdx = lChildren  # 父节点更新，进入下一层
                else:
                    value -= self.tree[lChildren]  # value大于左子节点，减去左子节点的数值后，遍历右子树
                    parentIdx = rChileren  # 父节点更新， 进入下一层
        # 将sumTree索引转为transition索引
        tranDataIndex = leafIndex - (self.bufferSize - 1)

        return leafIndex, self.tree[leafIndex], self.Transition[tranDataIndex]

    ## 根节点数值，即所有优先级总和
    def total_priority(self):
        return self.tree[0]


'''
定义经验回放技术类
'''


class Memory():
    def __init__(self, bufferSize):
        self.tree = SumTree(bufferSize)  # 创建一个Sum-Tree实例
        self.counter = 0  # 经验回放池中数据条数
        self.epsilon = 0.01  # 正向偏移以避免优先级为0
        self.alpha = 0.6  # [0,1],优先级使用程度系数
        self.beta = 0.4  # 初始IS值
        self.delta_beta = 0.001  # beta增加的步长
        self.absErrUpper = 1.  # TD误差绝对值的上界

    ## 往经验回放池中存入一条新的经验数据
    def store(self, newData):
        max_priority = np.max(self.tree.tree[-self.tree.bufferSize:])  # tree[-n:]表示倒数n个数, 遍历
        if max_priority == 0:  # 设置首条数据优先级为优先级上界
            max_priority = self.absErrUpper  # 设置新数据优先级为当前最大优先级
        self.tree.add(max_priority, newData)
        self.counter += 1

    ## 从经验回放池中取出batchSize个数据
    def sample(self, batchSize):
        # indexes储存取出的优先级在sumTree中索引，一维向量
        # samples存储取出的经验数据，二维矩阵
        # ISWeight储存权重，一维向量
        indexes, samples, ISWeights = np.empty(batchSize, dtype=np.int32), \
                                      np.empty((batchSize, self.tree.Transition[0].size)), \
                                      np.empty(batchSize)
        # 将优先级总和batchSize等分
        pri_seg = self.tree.total_priority() / batchSize
        # IS值逐渐增加到1，然后保持不变
        self.beta = np.min([1., self.beta + self.delta_beta])
        # 最小优先级占总优先级之比
        min_prob = np.min(self.tree.tree[-self.tree.bufferSize:]) / self.tree.total_priority()
        # 修正最小优先级占总优先级之比，当经验回放池未满和优先级为0时会用上
        if min_prob == 0:
            min_prob = 0.00001
        # 取出batchSize个数据
        for i in range(batchSize):
            a, b = pri_seg * i, pri_seg * (i + 1)  # 第i段优先级区间
            value = np.random.uniform(a, b)  # 在第i段优先级区间随机生成一个数
            # 返回sumTree中索引，优先级数值，对应的经验数据
            index, priority, sampleData = self.tree.getLeaf(value)
            # 抽样出的优先级占总优先级之比
            prob = priority / self.tree.total_priority()
            # 计算权重
            ISWeights[i] = np.power(prob / min_prob, -self.beta)
            indexes[i], samples[i, :] = index, sampleData

        return indexes, samples, ISWeights

    ## 调整批量数据
    def batchUpdate(self, sumTreeIndexes, absErrors):
        absErrors += self.epsilon  # 加上一个正向偏移，避免为0
        # TD误差绝对值不要超过上界
        clipped_errors = np.minimum(absErrors, self.absErrUpper)
        # alpha决定在多大程度上使用优先级
        prioritys = np.power(clipped_errors, self.alpha)
        # 更新优先级，同时更新树
        for index, priority in zip(sumTreeIndexes, prioritys):
            self.tree.update(index, priority)


'''
定义Q-网络类
'''


class NN(nn.Module):  # 继承与torch的nn.module类
    ## 类构造函数
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()  # 将输入拉直成向量
        # 定义Q网络
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 20),  # 输入层到第1隐藏层的线性部分
            nn.ReLU(),  # 第1隐藏层激活函数
            nn.Linear(20, 20),  # 第1隐藏层到第2隐藏层的线性部分
            nn.ReLU(),  # 第2隐藏层激活函数
            nn.Linear(20, output_size)  # 第2隐藏层到输出层
        )

    def forward(self, x):  # 前向传播函数
        x = self.flatten(x)  # 将输入拉直成向量
        logits = self.linear_relu_stack(x)  # 前向传播，预测x的值
        return logits  # 返回预测值


'''
定义PER-DQN方法类
'''


class DqnWithPER():
    def __init__(self, env, epsilon=0.1, learningRate=0.01,
                 bufferSize=2000, batchSize=32, targetReplaceIter=100):
        self.replayBuffer = Memory(batchSize)  # 初始化经验回放池
        self.env = env  # 环境模型
        self.epsilon = epsilon  # epsilon-greedy策略的参数
        self.learning_rate = learningRate  # 学习率
        self.replaySize = bufferSize  # 经验回放池最大容量
        self.batchSize = batchSize  # 批量尺度
        self.targetReplaceIter = targetReplaceIter
        self.learnStepCounter = 0

        self.create_Q_network()  # 生成Q网络实体
        self.create_training_method()  # Q网络优化器

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
        state = torch.from_numpy(np.expand_dims(state, 0))  # 转换成张量 在axis=0维度上扩展，(x,y)->(1,x,y)
        state = state.to(torch.float32)  # 张量的默认类型为float32

        # 计算所有动作值
        Q_value = self.Q_network.forward(state)
        # 以epsilon设定动作概率
        A = np.ones(self.env.aspace_size) * self.epsilon / self.env.aspace_size
        # 选取最大动作值对应的动作
        best = np.argmax(Q_value.detach().numpy())  # Q_value不需要反向传播计算梯度，从当前计算图中分离出来，再转成ndarray
        # 以 1 - epsilon的概率设定贪婪动作
        A[best] += 1 - self.epsilon
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
        expData = np.hstack((state, one_hot_action, reward, next_state, done))
        self.replayBuffer.store(expData)
        # 经验回放池中数据量多于一个批量batch_size就可以开始训练Q网络
        if self.replayBuffer.counter > self.batchSize:
            self.trainQNetwork()

    # Q网络训练函数
    def trainQNetwork(self):
        # 从经验回放池中随机抽取一个批量,此处miniBatch是一个一维向量
        sumTreeIndexes, miniBatch, ISWeights = self.replayBuffer.sample(self.batchSize)
        # miniBatch：[state, action, reward, next_state, done]
        state_batch = miniBatch[:, 0:self.env.state_dim]
        action_batch = miniBatch[:, self.env.state_dim:self.env.state_dim + self.env.aspace_size]

        # 计算TD目标值
        y_batch = []
        for x in miniBatch:  # 对minibatch中每一条MDP数据循环
            if x[-1]:  # 如果已经到达终止状态
                y_batch.append(x[self.env.state_dim + self.env.aspace_size])  # 记录一个mdp的reward
            else:  # 尚未达到终止状态
                next_state = x[-self.env.state_dim - 1:-1]
                temp = torch.from_numpy(next_state).unsqueeze(0).to(torch.float32)
                value_next = self.Q_network_t(temp)
                td_target = x[self.env.state_dim + self.env.aspace_size] \
                            + self.env.gamma * torch.max(value_next)
                y_batch.append(td_target.item())  # item()表示从td_target张量中取出该元素值
        y_batch = np.array(y_batch)

        # 将numpy.array数据转换为torch.tensor数据
        state_batch = torch.from_numpy(state_batch).to(torch.float32)  # shape= 32 x 4
        action_batch = torch.from_numpy(action_batch).to(torch.float32)  # shape= 32 x 2
        y_batch = torch.from_numpy(y_batch).to(torch.float32)  # shape= 32 x 1

        self.Q_network.train()  # 声明训练过程
        # output = self.Q_network(state_batch)
        # output_shape = output.shape
        # action_batch_shape = action_batch.shape
        # print("Q_network output={}".format(output_shape))

        # 预测批量值和损失函数
        pred = torch.sum(torch.multiply(self.Q_network(state_batch), action_batch), dim=1)  # dim=1，表示张量只有一个维度，类似于一维数组

        # Importance-Sample权重设置
        ISWeights = torch.from_numpy(ISWeights).to(torch.float32)
        pred, y_batch = ISWeights*pred, ISWeights*y_batch
        loss = self.loss_fun(pred, y_batch)

        # 误差反向传播，训练Q-网络
        self.optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 求各个参数的梯度值
        self.optimizer.step()  # 误差反向传播修改参数

        # 计算被抽取数据TD误差绝对值
        absErrors = torch.abs(pred-y_batch).detach().numpy()
        # 更新被抽取数据的优先级
        self.replayBuffer.batchUpdate(sumTreeIndexes, absErrors)

    # 训练函数
    def train(self, num_episodes=400, num_steps=2000, render=False):
        # 外层循环指导最大轮次
        rewards = []  # 每一回合的累积奖励
        for episode in range(num_episodes):
            state = self.env.reset()  # 环境初始化

            # 内层循环指导最大交互次数或到达终止状态
            reward_sum = 0  # 当前轮次的累积奖励
            for step in range(num_steps):
                env.render()
                self.learnStepCounter += 1
                action = self.egreedy_action(state)  # epsilon-贪婪策略选定动作
                next_state, reward, done, _ = self.env.step(action)  # 交互一个时间步

                # modify the reward  修改奖励确实能加速训练效果的提升
                # x, x_dot, theta, theta_dot = next_state
                # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                # r = r1 + r2

                reward_sum += reward  # 累积折扣奖励

                self.perceive(state, action, reward, next_state, done)  # 经验回放技术，训练
                state = next_state  # 更新状态
                if (self.learnStepCounter + 1) % self.targetReplaceIter == 0:  # 目标Q网络参数更新
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
        filepath = 'train.png'
        plt.savefig(filepath, dpi=300)
        plt.show()

    # 测试函数
    def test(self, num_episodes=100, render=False):
        # 循环直到最大测试轮数
        rewards = []  # 每一回合的累积奖励
        for episode in range(num_episodes):
            state = self.env.reset()  # 环境状态初始化

            # 循环直到到达终止状态
            reward_sum = 0  # 当前轮次的累积奖励
            while True:
                if render:
                    env.render()
                action = self.egreedy_action(state)  # epsilon-贪婪策略选定动作
                next_state, reward, end, info = self.env.step(action)  # 交互一个时间步
                reward_sum += reward  # 累积奖励
                state = next_state  # 状态更新

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
        filepath = 'test.png'
        plt.savefig(filepath, dpi=300)
        plt.show()

        return score  # 返回测试得分


'''主程序'''
if __name__ == '__main__':
    # 加载环境
    env = gym.make('CartPole-v0')  # 导入CartPole环境
    env.gamma = 0.9  # 折扣系数
    env.state_dim = env.observation_space.shape[0]  # 获取行维度
    env.aspace_size = env.action_space.n  # 离散动作个数

    agent = DqnWithPER(env)  # 创建一个DQN2015智能体
    agent.train(render=True)  # 训练智能体
    agent.test(render=True)  # 测试智能体
