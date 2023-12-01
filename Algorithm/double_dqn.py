import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np


# from torch.optim.lr_scheduler import ExponentialLR
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """ 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        """
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


class MLP(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        """ 初始化q网络，为全连接网络
            n_states: 输入的特征数即环境的状态维度
            n_actions: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc4 = nn.Linear(hidden_dim, n_actions)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc4(x)


class DoubleDQN:
    def __init__(self, n_states, n_actions, cfg):
        self.n_actions = n_actions  # 总的动作个数
        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma  # 奖励的折扣因子
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-1. * frame_idx / cfg.frame_idx)
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(n_states, n_actions, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(n_states, n_actions, hidden_dim=cfg.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        # self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)  # 以指数方式调整学习率
        self.loss = 0
        self.memory = ReplayBuffer(cfg.memory_capacity)

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

        if len(self.memory) < self.batch_size:
            return
        # 从memory中随机采样transition
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # convert to tensor
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  # 例如tensor([[1],...,[0]])
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  # tensor([1., 1.,...,1])
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)  # 将bool转为float然后转为张量
        # 计算当前(s_t,a)对应的Q(s_t, a)
        q_values = self.policy_net(state_batch)
        next_q_values = self.policy_net(next_state_batch)
        # 代入当前选择的action，得到Q(s_t|a=a_t)
        q_value = q_values.gather(dim=1, index=action_batch)
        '''以下是Nature DQN的q_target计算方式
        # 计算所有next states的Q'(s_{t+1})的最大值，Q'为目标网络的q函数
        next_q_state_value = self.target_net(next_state_batch).max(1)[0].detach()  
        '''
        '''以下是Double DQN q_target计算方式'''
        next_target_values = self.target_net(next_state_batch)
        # 选出Q(s_t‘, a)对应的action，代入到next_target_values获得target net对应的next_q_value，即Q’(s_t|a=argmax Q(s_t‘, a))
        next_target_q_value = next_target_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        q_target = reward_batch + self.gamma * next_target_q_value * (1 - done_batch)
        self.loss = nn.MSELoss()(q_value, q_target.unsqueeze(1))  # 计算 均方误差loss
        # 优化模型
        self.optimizer.zero_grad()  # zero_grad清除上一步所有旧的gradients from the last step
        # loss.backward()使用backpropagation计算loss相对于所有parameters(需要gradients)的微分
        self.loss.backward()
        # self.optimizer.step()  # 更新模型
        # self.scheduler.step()  # 更新学习

        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()  # 更新模型
        # self.scheduler.step()  # 更新学习率

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'DDQN_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'DDQN_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
