# TODO 添加电磁干扰
# TODO 改写干扰具体实现方式
# TODO 天线电偶极子干扰模型

import random
import numpy as np
import math
import gym


class UAVEnv(object):
    uav_h = 100  # 无人机高度    100unit,each 10meters
    v_uav = 5  # 无人机飞行速度
    square_length = 100
    uav_range = 20
    c = 3e8  # 光速
    B = 1e6  # 带宽 1MHZ
    fc = 2e9  # 载波频率
    pt = 10 ** ((20 - 30) / 10)  # 20dBm
    sigma = 10 ** ((-90 - 30) / 10)  # 加性高斯白噪声功率
    beta = 0.8  # 历史影响因子
    # pt_emi = 10**((-10 - 30)/10)  #3db
    pt_emi = 10 ** ((1) / 10)  # 3db
    a = 9.61
    b = 0.28
    sitaL = 1
    sitaNL = 20
    thru = 1458.6969302811174  # 水平10m内算成功
    # thru = 1500
    # thru = 32061593.633803133  # 水平10m内算成功
    # thru = 4e7
    # 1Mbps = 1e6bps
    #  200Kbps =2e5bps
    A = sitaL - sitaNL

    # 10 ^ (0.1) * ((3 * 10 ^ 8) / (4 * pi * ((x - 50) ^ 2 + (y - 50) ^ 2 + (100) ^ 2) ^ (1 / 2))) ^ 2

    raw = []
    with open('settings/u_loc.txt', 'r') as f:
        for line in f:
            raw.append(line.split())
    myULoc = np.array(raw).astype(float)
    N_U = len(myULoc)  # 埋压人员数量

    def __init__(self):
        self.u_loc = self.myULoc
        self.state = None
        self.bps = np.zeros((len(self.u_loc), 1))  # 传输速率
        self.bps_pr = np.zeros((len(self.u_loc), 1))
        self.connect = np.zeros((len(self.u_loc), 1))
        self.n_actions = 4
        self.n_states = 2 + 2
        self.time = 1

    def step(self, action):
        tmp_pr = np.zeros((len(self.u_loc), 1))
        throughtout = np.zeros((len(self.u_loc), 1))
        sinr = np.zeros((len(self.u_loc), 1))
        g = np.zeros((len(self.u_loc), 1))
        pl = np.zeros((len(self.u_loc), 1))
        tmp = np.zeros((len(self.u_loc), 1))
        dis = np.zeros((len(self.u_loc), 1))
        sita = np.zeros((len(self.u_loc), 1))
        bps_pr = np.zeros((len(self.u_loc), 1))
        num = 0
        step_reward = -1
        out_reward = 0
        count_reward = 0
        finish_reward = 0
        connect_reward = 0
        x_u_pr, y_u_pr, c_pr, bps_max_pr = self.state

        done = False
        if action == 0:
            x_u = x_u_pr
            y_u = y_u_pr + self.v_uav
        if action == 1:
            x_u = x_u_pr
            y_u = y_u_pr - self.v_uav
        if action == 2:
            x_u = x_u_pr - self.v_uav
            y_u = y_u_pr
        if action == 3:
            x_u = x_u_pr + self.v_uav
            y_u = y_u_pr
        # TODO 电偶极子干扰模型
        d = math.sqrt((x_u - 50) ** 2 + (y_u - 50) ** 2 + self.uav_h ** 2)
        pr_emi = self.pt_emi * ((self.c / self.fc / (4 * math.pi * d)) ** 2)

        for i in range(len(self.u_loc)):
            tmp[i] = math.sqrt((self.u_loc[i][0] - x_u) ** 2 + (self.u_loc[i][1] - y_u) ** 2)
            if self.connect[i] == 1:
                continue
            if tmp[i] <= self.uav_range:
                num = num + 1

        if x_u > 100 or y_u > 100 or x_u < 0 or y_u < 0:
            done = True
            out_reward = -100

        # 无人机与地面用户距离以及通信速率
        for i in range(len(self.u_loc)):
            tmp[i] = math.sqrt((self.u_loc[i][0] - x_u) ** 2 + (self.u_loc[i][1] - y_u) ** 2)
            if self.connect[i] == 1:
                continue
            dis[i] = math.sqrt(tmp[i] ** 2 + self.uav_h ** 2)
            sita[i] = np.arctan(self.uav_h / tmp[i])
            pl[i] = self.A / (1 + self.a * np.exp((-self.b) * ((180 * sita[i] / math.pi) - self.a))) + 20 * np.log10(
                dis[i]) + 20 * np.log10(4 * math.pi * self.fc / 3e8) + self.sitaNL
            g[i] = 10 ** (-pl[i] / 10)
            if num == 0:
                sinr[i] = g[i] * self.pt / ((self.sigma * self.B) + pr_emi)
                self.bps[i] = self.B * math.log2(1 + sinr[i])
                self.connect[i] = 0
            else:
                # TODO 增加电磁干扰影响信噪比的噪声功率
                sinr[i] = g[i] * self.pt / (self.sigma * (self.B / num) + pr_emi)
                self.bps[i] = (self.B / num) * math.log2(1 + sinr[i])
            if self.bps[i] >= self.thru:
                self.connect[i] = 1
        nonzero_indices = np.nonzero(self.connect)  # 找到arr1中非零元素的索引
        corresponding_values = self.bps[nonzero_indices]  # 获取arr2中与arr1非零元素位置对应的值
        if len(corresponding_values) == 0:
            bps_max = 0
        else:
            bps_max = np.max(corresponding_values)  # 找到corresponding_values中的最大值
        c = np.sum(self.connect)
        if c == self.N_U:
            done = True
            finish_reward = 100
        self.state = x_u, y_u, c, bps_max
        count_reward = (c - c_pr)

        # reward = 0.3*count_reward + 0.2*out_reward + 0.1*finish_reward + 0.3*step_reward
        reward = 0.6 * count_reward + 0.2 * out_reward + 0.1 * finish_reward + 0.1 * step_reward
        # print(reward)
        # print(self.state[0, :2])
        return np.copy(self.state), reward, done, c

    def reset(self):
        x_u = 10
        y_u = 0
        num = 0
        tmp_pr = np.zeros((len(self.u_loc), 1))
        self.bps = np.zeros((len(self.u_loc), 1))  # 传输速率
        self.bps_pr = np.zeros((len(self.u_loc), 1))
        self.connect = np.zeros((len(self.u_loc), 1))
        throughtout = np.zeros((len(self.u_loc), 1))
        sinr = np.zeros((len(self.u_loc), 1))
        g = np.zeros((len(self.u_loc), 1))
        pl = np.zeros((len(self.u_loc), 1))
        tmp = np.zeros((len(self.u_loc), 1))
        dis = np.zeros((len(self.u_loc), 1))
        sita = np.zeros((len(self.u_loc), 1))

        # TODO 电偶极子干扰模型
        d = math.sqrt((x_u - 50) ** 2 + (y_u - 50) ** 2 + self.uav_h ** 2)
        pr_emi = self.pt_emi * ((self.c / self.fc / (4 * math.pi * d)) ** 2)

        for i in range(len(self.u_loc)):
            tmp[i] = math.sqrt((self.u_loc[i][0] - x_u) ** 2 + (self.u_loc[i][1] - y_u) ** 2)
            if tmp[i] < self.uav_range:
                num = num + 1
        # 无人机与地面用户距离以及通信速率
        for i in range(len(self.u_loc)):
            if self.connect[i] == 1:
                continue
            tmp[i] = math.sqrt((self.u_loc[i][0] - x_u) ** 2 + (self.u_loc[i][1] - y_u) ** 2)
            dis[i] = math.sqrt(tmp[i] ** 2 + self.uav_h ** 2)
            sita[i] = np.arctan(self.uav_h / tmp[i])
            pl[i] = self.A / (1 + self.a * np.exp((-self.b) * ((180 * sita[i] / math.pi) - self.a))) + 20 * np.log10(
                dis[i]) + 20 * np.log10(4 * math.pi * self.fc / 3e8) + self.sitaNL
            g[i] = 10 ** (-pl[i] / 10)
            if num == 0:
                sinr[i] = g[i] * self.pt / ((self.sigma * self.B) + pr_emi)
                self.bps[i] = (self.B) * math.log2(1 + sinr[i])
            else:
                sinr[i] = g[i] * self.pt / (self.sigma * (self.B / num) + pr_emi)
                self.bps[i] = (self.B / num) * math.log2(1 + sinr[i])
            if self.bps[i] >= self.thru:
                self.connect[i] = 1
        c = np.sum(self.connect)
        nonzero_indices = np.nonzero(self.connect)  # 找到arr1中非零元素的索引
        corresponding_values = self.bps[nonzero_indices]  # 获取arr2中与arr1非零元素位置对应的值
        if len(corresponding_values) == 0:
            bps_max = 0
        else:
            bps_max = np.max(corresponding_values)  # 找到corresponding_values中的最大值
        self.bps_pr = self.bps
        self.state = x_u, y_u, c, bps_max
        return np.copy(self.state)

    # def seed(self, seed=11):
    #     np.random.seed(seed)
