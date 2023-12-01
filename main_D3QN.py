# from lib2to3.pytree import type_repr
import sys
import os
# from parso import parse
import torch.nn as nn
import torch.nn.functional as F
import gym
import torch
import datetime
import numpy as np
import argparse
from utils import save_results_1, make_dir
from utils import plot_rewards, save_args, plot_ma_rewards, plot_allrewards, plot_sign, plot_steps,save_state
from Algorithm.dqn import DQN
from Algorithm.double_dqn import DoubleDQN
from Algorithm.dueling_dqn import DuelingDQN
from Algorithm.D3QN import D3QN
from uav_envs19_with_antenna_EMI import UAVEnv


curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径


def get_args():
    """ Hyper parameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--sign', default='the number of users', type=str, help='人数')
    parser.add_argument('--algo_name', default='D3QN', type=str, help="算法名称")
    parser.add_argument('--env_name', default='UAV-v1', type=str, help="环境名称")
    parser.add_argument('--train_eps', default=1000, type=int, help="训练回合数")
    parser.add_argument('--test_eps', default=50, type=int, help="测试回合数")
    parser.add_argument('--gamma', default=0.9, type=float, help="折扣因子")
    parser.add_argument('--epsilon_start', default=0.9, type=float, help="贪婪因子初始值")
    parser.add_argument('--epsilon_end', default=0.00001, type=float, help="贪婪因子终值")
    parser.add_argument('--frame_idx', default=10000, type=int, help="贪婪因子衰减率")
    parser.add_argument('--lr', default=0.0001, type=float, help="学习率")
    parser.add_argument('--memory_capacity', default=2000000, type=int, help="经验池容量")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--target_update', default=2, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--result_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                 '/' + curr_time + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                '/' + curr_time + '/models/')  # path to save models
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU
    return args


def env_agent_config(cfg, seed=1):
    """ 创建环境和智能体
    """

    # env = gym.make(cfg.env_name)  # 创建环境
    env = UAVEnv()  # 创建环境
    n_actions = env.n_actions  # 动作维度
    n_states = env.n_states  # 状态维度
    print(f"n_states: {n_states}, n_actions: {n_actions}")
    # agent = DQN(n_states, n_actions, cfg)       # 创建智能体
    # agent = DoubleDQN(n_states,n_actions,cfg)
    agent = D3QN(n_states,n_actions,cfg)
    if seed != 0:  # 设置随机种子
        torch.manual_seed(seed)
        # env.seed(seed)
        np.random.seed(seed)
    return env, agent


def train(cfg, env, agent):
    """ Training
    """
    print('Start training!')
    print(f'Env:{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    state_list = []
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = []      # 记录每回合的步数
    ma_step = []    # 记录每回合的滑动步数
    numlist = []    # 记录每回合滑动的人数
    numep_list = []     # 记录每回合的人数
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态

        while True:
            ep_step += 1
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done, num = env.step(action)  # 更新环境，返回transition
            agent.memory.push(state, action, reward, next_state, done)  # 保存transition
            state = next_state  # 更新下一个状态
            # state_list.append(state[:2])
            agent.update()  # 更新智能体
            ep_reward += reward  # 累加奖励
            ep_reward = ep_reward
            numep_list.append(num)
            if ep_step > 300:
                done = True
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:  # 智能体目标网络更新
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

        if numlist:
            numlist.append(0.9 * numlist[-1] + 0.1 * numep_list[-1])
        else:
            numlist.append(numep_list[-1])

        if ma_step:
            ma_step.append(0.9 * ma_step[-1] + 0.1 * ep_step)
        else:
            ma_step.append(ep_step)

        if (i_ep + 1) % 1 == 0:
            # save_state(state_list, 'D3QN')
            state_list = []
            print(
                f'Episode：{i_ep + 1}/{cfg.train_eps}, Reward:{ep_reward:.2f}, Step:{ep_step:.2f} , Epislon:{agent.epsilon(agent.frame_idx):.3f} , num:{numep_list[-1]:.2f}')
    print('Finish training!')
    return {'rewards': rewards, 'ma_rewards': ma_rewards, 'steps': steps, 'ma_step': ma_step, 'num': numlist}


def test(cfg, env, agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = []
    ma_step = []
    numlist = []
    numep_list = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            ep_step += 1
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done, num = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            ep_reward = ep_reward
            numep_list.append(num)
            if ep_step > 300:
                done = True
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)

        if numlist:
            numlist.append(0.9 * numlist[-1] + 0.1 * numep_list[-1])
        else:
            numlist.append(numep_list[-1])

        if ma_step:
            ma_step.append(0.9 * ma_step[-1] + 0.1 * ep_step)
        else:
            ma_step.append(ep_step)
        print(f'Episode：{i_ep + 1}/{cfg.test_eps}, Reward:{ep_reward:.2f}, Step:{ep_step:.2f}, num:{numep_list[-1]:.2f}')
    print('完成测试！')
    return {'rewards': rewards, 'ma_rewards': ma_rewards, 'steps': steps, 'ma_step': ma_step, 'num': numlist}


if __name__ == "__main__":
    cfg = get_args()
    # 训练
    env, agent = env_agent_config(cfg)
    res_dic = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)  # 创建保存结果和模型路径的文件夹
    save_args(cfg)
    agent.save(path=cfg.model_path)  # 保存模型
    save_results_1(res_dic, tag='train', path=cfg.result_path)  # 保存结果
    plot_ma_rewards(res_dic['ma_rewards'], cfg, tag="train")  # 画出结果
    plot_sign(res_dic['num'], cfg, tag="train")
    plot_steps(res_dic['ma_step'], cfg, tag="train")
    # 测试
    env, agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)  # 导入模型
    res_dic = test(cfg, env, agent)
    save_results_1(res_dic, tag='test', path=cfg.result_path)  # 保存结果
    plot_allrewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="test")  # 画出结果
    plot_sign(res_dic['num'], cfg, tag="test")
    plot_steps(res_dic['ma_step'], cfg, tag="test")