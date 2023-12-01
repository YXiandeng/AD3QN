import os
import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.font_manager import FontProperties  # 导入字体模块


def chinese_font():
    """
    设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
    """
    try:
        font = FontProperties(
            fname='/System/Library/Fonts/STHeiti Light.ttc', size=15)  # fname系统字体路径，此处是mac的
    except:
        font = None
    return font


def plot_rewards_cn(rewards, ma_rewards, cfg, tag='train'):
    """
    中文画图
    """
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(cfg.env_name, cfg.algo_name), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励', u'滑动平均奖励',), loc="best", prop=chinese_font())
    if cfg.save:
        plt.savefig(cfg.result_path + f"{tag}_rewards_curve_cn")
    plt.show()


def plot_allrewards(rewards, ma_rewards, cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_all_rewards_curve".format(tag))
    plt.show()


def plot_sign(sign, cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(sign, label='{}'.format(cfg.sign))
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_sign_curve".format(tag))
    plt.show()


def plot_steps(steps, cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(steps, label='step')
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_steps_curve".format(tag))
    plt.show()


def plot_rewards(rewards, cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_rewards_curve".format(tag))
    plt.show()


def plot_ma_rewards(ma_rewards, cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_ma_rewards_curve".format(tag))
    plt.show()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve")
    plt.show()


def save_results_1(dic, tag='train', path='./results'):
    """
    保存奖励
    """
    for key, value in dic.items():
        np.save(path + '{}_{}.npy'.format(tag, key), value)
    print('Results saved！')


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    """
    保存奖励
    """
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('Result saved!')


def make_dir(*paths):
    """
    创建文件夹
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def del_empty_dir(*paths):
    """
    删除目录下所有空文件夹
    """
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))


def save_args(args):
    # save parameters
    argsDict = args.__dict__
    with open(args.result_path + 'params.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    print("Parameters saved!")


# 保存飞行路径
def save_state(data, model):
    try:
        with open(f'settings/station_{model}.txt', 'a') as f:
            f.write("**********\n")  # 插入星星
            for state in data:
                f.write(f'{state[0]}\t{state[1]}\n')
            f.write("**********\n")
    except Exception as e:
        print("写入数据时发生错误：", e)


# Greedy-epsilon
class EpsilonGreedy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)


# Ornstein-Ulhenbeck Process
class OUNoise:
    def __init__(self, mean=0.0, mean_attraction_constant=0.6, variance=0.6, decay_rate=5e-4):
        self.mean = mean
        self.mean_attraction_constant = mean_attraction_constant
        self.variance = variance
        self.variance_min = 0
        self.decay_rate = decay_rate
        self.action_dim = 10
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mean

    def evolve_state(self, t):
        x = self.state
        dx = self.mean_attraction_constant * (self.mean - x) * t + self.variance * np.random.randn(
            self.action_dim) * math.sqrt(t)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state(t)
        decayed_variance = self.variance * (1 - self.decay_rate)
        self.variance = max(decayed_variance, self.variance_min)
        return np.clip(action + ou_state, self.variance_min, self.variance)
