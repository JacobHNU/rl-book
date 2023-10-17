"""
Actor-Critic Policy Gradient
1. 初始化
    策略参数θ
    价值参数ω
2. 外层循环 episode==1~num_episodes
    初始化环境状态 s=s0
    内层循环  到达终止状态s==end
        选择并执行动作  a=π(s;θ)
        环境状态转移并反馈即时奖励 r,s',end,_ = env.step(a)
        预测下一个动作  a'=π(s';θ)
        价值网络打分，计算(s,a)的预测值 y'=Q(s,a;ω)
        计算TD目标值
            y= r              if end=True
            y=r+γQ(s',a';ω)  if end=False
        更新策略/价值参数 θ<-θ’ , ω<-ω'
        更新状态  s<-s'
3. 输出：
    最优策略参数θ*，最有价值参数ω*
"""
'''
REINFORCE 与 Actor-Critic 的区别
REINFORCE算法用了状态价值函数来作为基准函数，来降低梯度估计的方差。
Actor-Critic也是类似的想法，只不过是actor还是用来评估策略的同时，利用了critic网络来评估状态价值函数，
    同时计算上将 回报G-B(s)，利用时序差分估计来代替，这样就可以用价值（状态价值函数）估计来估计回报
'''

import gym
import numpy as np
from torch import nn
import torch
import matplotlib.pyplot as plt


'''
定义Actor网络， 策略网络
'''
class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor,self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # 定义策略网络层
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(input_size,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,output_size),
            nn.Softmax(dim=-1)
        )

