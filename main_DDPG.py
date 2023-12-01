#!/usr/bin/env python
# coding=utf-8
import sys, os

curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

import datetime
import gym
import torch
import argparse

from OUnoise import OUNoise
from Algorithm.ddpg import DDPG
from utils import save_results_1, make_dir
from utils import plot_rewards,save_args,plot_ma_rewards,plot_allrewards,plot_sign,plot_steps
from myGym.researchGym.envs.uav_communication_envs import UAVEnv

def get_args():
    """ Hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--sign', default='the number of users', type=str, help='人数')
    parser.add_argument('--algo_name', default='DDPG', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='UAVEnv-v1', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=1000, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=50, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.9, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=1e-3, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-3, type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=100000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--target_update', default=2, type=int)
    parser.add_argument('--soft_tau', default=1e-2, type=float)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--result_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                 '/' + curr_time + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                '/' + curr_time + '/models/')  # path to save models
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # check GPU
    return args


def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env_name)  # 装饰action噪声
    env.seed(seed)  # 随机种子
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    print(f"n_states: {n_states}, n_actions: {n_actions}")
    agent = DDPG(n_states, n_actions, cfg)
    return env, agent


def train(cfg, env, agent):
    print('Start training!')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    ou_noise = OUNoise(env.action_space)  # noise of action
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = []      # 记录每回合的步数
    ma_step = []    # 记录每回合的滑动步数
    numlist = []    # 记录每回合滑动的人数
    numep_list = []     # 记录每回合的人数
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            action = ou_noise.get_action(action, i_step)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            num = info['my_float_info']
            numep_list.append(num)
            if i_step>300:
                done=True
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        if (i_ep + 1) % 1 == 0:
            print(
                f'Episode：{i_ep + 1}/{cfg.train_eps}, Reward:{ep_reward:.2f}, Step:{i_step:.2f} , num:{numep_list[-1]:.2f}')
        rewards.append(ep_reward)
        steps.append(i_step)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

        if numlist:
            numlist.append(0.9 * numlist[-1] + 0.1 * numep_list[-1])
        else:
            numlist.append(numep_list[-1])

        if ma_step:
            ma_step.append(0.9 * ma_step[-1] + 0.1 * i_step)
        else:
            ma_step.append(i_step)
    print('Finish training!')
    return {'rewards': rewards, 'ma_rewards': ma_rewards, 'steps': steps, 'ma_step': ma_step, 'num': numlist}


def test(cfg, env, agent):
    print('Start testing')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = []
    ma_step = []
    numlist = []
    numep_list = []
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            num = info['my_float_info']
            numep_list.append(num)
            if i_step>300:
                done=True
            state = next_state
        rewards.append(ep_reward)
        steps.append(i_step)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

        if numlist:
            numlist.append(0.9 * numlist[-1] + 0.1 * numep_list[-1])
        else:
            numlist.append(numep_list[-1])

        if ma_step:
            ma_step.append(0.9 * ma_step[-1] + 0.1 * i_step)
        else:
            ma_step.append(i_step)
        print(f'Episode：{i_ep + 1}/{cfg.test_eps}, Reward:{ep_reward:.2f}, Step:{i_step:.2f}, num:{numep_list[-1]:.2f}')
    print('Finish testing!')
    return {'rewards': rewards, 'ma_rewards': ma_rewards, 'steps': steps, 'ma_step': ma_step, 'num': numlist}


if __name__ == "__main__":
    cfg = get_args()
    # training
    env, agent = env_agent_config(cfg, seed=1)
    res_dic = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    save_args(cfg)
    agent.save(path=cfg.model_path)
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
