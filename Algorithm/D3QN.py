#!/usr/bin/env python
# coding=utf-8
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
# from torch.optim.lr_scheduler import ExponentialLR


class D3QNNet(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super(D3QNNet,self).__init__()
        # 隐藏层
        self.hidden = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU()
        )

        # 优势函数
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

        # 价值函数
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.hidden(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage - advantage.mean()

    # def act(self, state, epsilon):
    #     if random.random() > epsilon:
    #         with torch.no_grad():
    #             state   = Variable(torch.FloatTensor(state).unsqueeze(0))
    #             q_value = self.forward(state)
    #             action  = q_value.max(1)[1].item()
    #     else:
    #         action = random.randrange(env.action_space.n)
    #     return action

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # 随机采出小批量转移
        state, action, reward, next_state, done = zip(*batch)  # 解压成状态，动作等
        return state, action, reward, next_state, done

    def __len__(self):
        """ 返回当前存储的量
        """
        return len(self.buffer)


class D3QN:
    def __init__(self, n_states, n_actions, cfg):

        self.n_actions = n_actions  # 总的动作个数
        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma  # 奖励的折扣因子
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.frame_idx)
        self.batch_size = cfg.batch_size
        self.policy_net = D3QNNet(n_states, n_actions, cfg.hidden_dim).to(self.device)
        self.target_net = D3QNNet(n_states, n_actions, cfg.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)  # Adam优化器
        # self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)  # 以指数方式调整学习率
        self.memory = ReplayBuffer(cfg.memory_capacity)  # 经验回放

    def choose_action(self, state):
        """选择动作
        """
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = torch.argmax(q_values)
        else:
            action = random.randrange(self.n_actions)  # 随机选择一个动作
        return action

    def update(self):
        if len(self.memory) < self.batch_size:  # 当memory中不满足一个批量时，不更新策略
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float).squeeze(1)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float).squeeze(1)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        # DDQN的loss的计算方法
        # 计算当前(s_t,a)对应的Q(s_t, a)
        q_values = self.policy_net(state_batch)
        next_q_values = self.policy_net(next_state_batch)
        # 代入当前选择的action，得到Q(s_t|a=a_t)
        q_value = q_values.gather(dim=1, index=action_batch)
        next_target_values = self.target_net(next_state_batch)
        # 选出Q(s_t‘, a)对应的action，代入到next_target_values获得target net对应的next_q_value，即Q’(s_t|a=argmax Q(s_t‘, a))
        next_target_q_value = next_target_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        q_target = reward_batch + self.gamma * next_target_q_value * (1-done_batch)
        loss = nn.MSELoss()(q_value, q_target.unsqueeze(1))  # 计算 均方误差loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()  # 更新学习率

        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # self.scheduler.step()  # 更新学习率

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'D3QN_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'D3QN_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
