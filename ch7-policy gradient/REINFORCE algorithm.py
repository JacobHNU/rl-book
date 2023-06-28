'''
REINFORCE 算法

在前面章节中，从动作价值函数导出最优策略估计往往有特定的形式（epsilon-Greedy策略）。
与之相比，从动作偏好导出的最优策略的估计不拘泥于特定的形式，其每个动作都可以有不同的概率值。
采用迭代的方法更新参数θ，则 π(a|s;θ)可以逼近确定性策略，不需要调节epsilon等参数。
'''


import gym
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


# 离散动作空间，用softmax
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN,self).__init__()
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
        output = nn.functional.softmax(features, dim=-1)

        return output

class  REINFORCE():
    def __init__(self, env):
        self.env = env
        self.aspace = np.arange(self.env.aspace_size)
        self.P_net = NN(self.env.state_dim, self.env.aspace_size)
        self.opt = torch.optim.Adam(self.P_net.parameters(), lr=1e-2)

    ## 计算折扣奖励
    def discount_rewards(self, rewards):
        r = np.array([self.env.gamma**i*rewards[i] for
                      i in range(len(rewards))])               # r =r + gamma**i*rewards[i]
        r = r[::-1].cumsum()[::-1]   # 至后向前依次计算累积折扣奖励

        return r-r.mean()   # 降低学习过程中的方差，引入基线函数

    def train(self, num_episodes=1000, batch_size=10):
        total_rewards = []   # 存放回报数据
        batch_rewards = []   # 存放批量回报数据
        batch_actions = []   # 存放批量动作数据
        batch_states = []    # 存放批量状态数据
        batch_counter = 1    # 初始化批量计数器

        # 外层循环直到最大训练轮数
        for ep in range(num_episodes):
            s = self.env.reset()
            states = []
            rewards = []
            actions = []
            end = False
            # 内层循环直到终止状态
            while end == False:
                env.render()
                prob = self.P_net(
                    torch.Tensor([s])).detach().squeeze().numpy()  # 近似的策略
                a = np.random.choice(self.aspace, p=prob)
                s_,r,end,_ = self.env.step(a)
                states.append(s_)
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
                    print('ep=', ep, 'total_rewards=', total_rewards[ep])
                    # 以batch_size回合的所有交互数据为一个批量
                    if batch_counter == batch_size:
                        state_tensor = torch.Tensor(batch_states)
                        reward_tensor = torch.Tensor(batch_rewards)
                        action_tensor = torch.Tensor(batch_actions)
                        # 损失函数
                        log_probs = torch.log(self.P_net(state_tensor))   # lnπ(a|s;θ)
                        selected_log_probs = reward_tensor * log_probs[    # 选择的动作的概率 r*lnπ(a|s;θ)
                            np.arange(len(action_tensor)), action_tensor.type(torch.long)]
                        loss = -selected_log_probs.mean()                 # 批量数据求期望，E[r*lnπ(a|s;θ)]

                        # 误差反向传播和训练
                        self.opt.zero_grad()   # 梯度归零
                        loss.backward()        # 求各个参数的梯度值
                        self.opt.step()        # 误差反向传播
                        # 数据初始化， 为下一个批量做准备
                        batch_actions, batch_rewards, batch_states = [], [], []
                        batch_counter = 1

        # 图示训练过程
        plt.figure('train')
        plt.title('train')
        window = 40
        smooth_r = [np.mean(total_rewards[i - window:i + 1]) if i > window
                    else np.mean(total_rewards[:i + 1])
                    for i in range(len(total_rewards))]
        plt.plot(total_rewards, label='accumulate rewards')
        plt.plot(smooth_r, label='smoothed accumulate rewards')
        plt.legend()
        filepath = 'train.png'
        plt.savefig(filepath, dpi=300)
        # plt.show()

        ## 测试函数

    def test(self, num_episodes=100):
        total_rewards = []  # 存放回报数据

        # 外层循环直到最大测试轮数
        for _ in range(num_episodes):
            rewards = []  # 存放即时奖励数据
            s = self.env.reset()  # 环境状态初始化
            # 内层循环直到到达终止状态
            while True:
                prob = self.P_net(
                    torch.Tensor([s])).detach().squeeze().numpy()
                action = np.random.choice(self.aspace, p=prob)
                s_, r, end, info = env.step(action)
                rewards.append(r)
                if end:
                    total_rewards.append(sum(rewards))
                    break
                else:
                    s = s_  # 更新状态，继续交互

        # 计算测试得分
        score = np.mean(np.array(total_rewards))

        # 图示测试结果
        plt.figure('test')
        plt.title('test: score=' + str(score))
        plt.plot(total_rewards, label='accumulate rewards')
        plt.legend()
        filepath = 'test.png'
        plt.savefig(filepath, dpi=300)
        # plt.show()

        return score  # 返回测试得分


'''
主程序
'''
if __name__ == '__main__':
    # 导入环境
    #    env = gym.make('Acrobot-v1')  # 'Acrobot-v1','CartPole-v0'

    # import MountainCar
    #
    # env = MountainCar.MountainCarEnv()

    env = gym.make('CartPole-v0')
    env.gamma = 0.99  # 补充定义折扣系数
    env.state_dim = env.observation_space.shape[0]  # 状态维度
    env.aspace_size = env.action_space.n  # 离散动作个数

    for i in range(10):
        print('第%d次训练' % i)
        agent = REINFORCE(env)  # 创建一个REINDOECE类智能体
        agent.train()  # 训练
        agent.test()  # 测试








