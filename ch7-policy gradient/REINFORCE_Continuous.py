# -*- coding: utf-8 -*-
"""
带基线的简单策略梯度算法 REINFORCE with baseline
    为了降低学习过程中的方差，引入基线函数B(s)
算法描述：
    输入：环境
    输出： 最优策略的估计
    1. （初始化） 参数初始化 π(A|S;θ)中的θ
    2. （回合更新）对每个回合执行以下操作
        2.1 （采样）用策略 π(θ)生成轨迹S_0,A_0,R_1,S_1,A_1,...,S_{T-1},A_{T-1},R_T,S_T
        2.2 （初始化回报） G <- 0
        2.3 对 t=T-1,T-2,...,0，执行以下步骤：
            2.3.1 （更新回报）G <- γG + R_{t+1}
            2.3.2 （更新价值）G <- G-R.mean()
            2.3.3 （更新策略） 更新θ以减小 -γ^{t}[G-R.mean()] lnπ(A_t|S_t;θ)
"""
import gym
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

'''
定义为训练均值和方差的网络
'''
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20, output_size)
            )

    # 前向传播函数
    def forward(self,x):
        x = self.flatten(x)
        features = self.linear_relu_stack(x)
        mu_sigma = nn.functional.softmax(features, dim=-1)

        return mu_sigma                               # 输出均值和方差

'''
REINFORCE策略梯度法类
'''
class REINFORCE():
    def __init__(self, env):
        self.env = env
        self.P_net = NN(self.env.state_dim, 2)
        self.opt = torch.optim.Adam(self.P_net.parameters(), lr=1e-2)

    ## 评价函数
    def log_prob(self, mus, sigmas, xs):
        # 内置高斯函数
        def gaussian(mu, sigma, x):
            temp1 = 1.0 / (np.sqrt(2.0*np.pi)*sigma)
            temp2 = -(x-mu)**2/(2.0*sigma**2)
            return temp1*torch.exp(temp2)

        # 计算各样本点的高斯函数值
        res = [gaussian(mu, sigma, x) for mu, sigma, x in zip(mus,sigmas,xs)]
        return torch.Tensor(res)

    ## 计算累积折扣奖励
    def discount_rewards(self, rewards):
        r = np.array([self.env.gamma**i*rewards[i] for
                      i in range(len(rewards))])
        r = r[::-1].cumsum()[::-1]   # 从后往前依次计算累积折扣奖励

        return r - r.mean()  # 以累积折扣奖励的均值作为基线函数

    def train(self, num_episode=50, batch_size=10):
        total_rewards = []    # 存放回报数据
        batch_rewards = []    # 存放批量回报数据
        batch_actions = []    # 存放批量动作数据
        batch_states = []     # 存放批量状态数据
        batch_counter = 1     # 初始化批量计数器

        # 外层循环直到最大训练轮数
        for ep in range(num_episode):
            s = self.env.reset()
            states = []
            rewards = []
            actions = []
            end = False
            # 内层循环直到终止状态
            while end == False:
                mu, sigma = self.P_net(
                    torch.Tensor([s])).detach().squeeze().numpy()
                a = np.random.normal(mu, sigma)
                s_, r, end, _ = self.env.step(a)
                states.append(s)
                rewards.append(r)
                actions.append(a)
                s = s_
                if end:
                    # 将新得到的数据加入到批量数据中
                    batch_rewards.extend(self.discount_rewards(rewards))
                    batch_states.extend(states)
                    batch_actions.extend(actions)
                    batch_counter += 1
                    # 计算当前轮次的回报
                    total_rewards.append(sum(rewards))
                    # 以batch_size回合的所有交互数据为一个批量
                    if batch_counter == batch_size:
                        state_tensor = torch.Tensor(batch_states)
                        reward_tensor = torch.Tensor(batch_rewards)
                        action_tensor = torch.LongTensor(batch_actions)
                        # 损失函数
                        mu_sigma = self.P_net(state_tensor)
                        log_probs = self.log_prob(mu_sigma[:,0],
                                                  mu_sigma[:,1],
                                                  action_tensor,)
                        selected_log_probs = reward_tensor * log_probs
                        selected_log_probs.requires_grad = True
                        loss = -selected_log_probs.mean()
                        # 误差反向传播
                        self.opt.zero_grad()  # 梯度归零
                        loss.backward()       # 求各个参数的梯度值
                        self.opt.step()       # 反向传播
                        # 数据初始化，为下一个批量做准备
                        batch_rewards,batch_actions,batch_states = [], [], []
                        batch_counter = 1

        # 图示训练过程
        plt.figure('train')
        plt.title('train')
        window = 10
        smooth_r = [np.mean(total_rewards[i-window:i+1]) if i > window
                        else np.mean(total_rewards[:i+1])
                        for i in range(len(total_rewards))]
        plt.plot(total_rewards,label='accumulate rewards')
        plt.plot(smooth_r,label='smoothed accumulate rewards')
        plt.legend()
        filepath = 'RENINFORCE_train.png'
        plt.savefig(filepath, dpi=300)
        plt.show()

    # 测试函数
    def test(self, num_episodes=100):
        total_rewards = []    # 存放回报数据

        # 外层循环指导最大测试轮数
        for _ in range(num_episodes):
            rewards = []   # 存放即时奖励
            s = self.env.reset()   # 环境状态初始化
            # 内层循环直到到达终止状态
            while True:
                mu, sigma = self.P_net(
                    torch.Tensor([s])).detach().squeeze().numpy()
                a = np.random.normal(mu,sigma)
                s_,r,end,info = self.env.step(a)
                rewards.append(r)
                if end:
                    total_rewards.append(sum(rewards))
                    break
                else:
                    s = s_    # 更新状态，继续交互

        # 计算测试得分
        score = np.mean(np.array(total_rewards))

        # 图数测试结果
        plt.figure('test')
        plt.title('test: score='+str(score))
        plt.plot(total_rewards, label='accumulate rewards')
        plt.legend()
        filepath = 'RENINFORCE_test.png'
        plt.savefig(filepath,dpi=300)
        plt.show()

        return score        # 返回测试得分


'''
主程序
'''
if __name__ == '__main__':
    import MountainCarContinious
    env = MountainCarContinious.MountainCarContinuousEnv()

    env.gamma = 0.99    # 补充定义折扣系数
    env.state_dim = env.observation_space.shape[0]  # 状态维度

    for i in range(10):
        print('第%d次训练'%i)
        agent = REINFORCE(env)   # 创建一个REINFORCE类智能体
        agent.train()            # 训练
        agent.test()             # 测试





