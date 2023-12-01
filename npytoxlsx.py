def savenpyasexcel(title, output):
    # title是要转换的文件位置，output是保存的文件位置
    import pandas as pd
    import numpy as np
    data = np.load(title)  # 读取numpy文件
    data_df = pd.DataFrame(data)  # 关键1，将ndarray格式转换为DataFrame
    # rows, cols = data.shape
    # # 更改表的索引
    # data_index = []
    # for i in range(rows):
    #     data_index.append(i)
    # data_df.index = data_index
    # # 更改表的索引
    # data_index = []
    # for i in range(cols):
    #     data_index.append(i)
    # data_df.index = data_index
    # data_df.columns = data_index

    # 将文件写入excel表格中
    writer = pd.ExcelWriter(output + '.xlsx')  # 关键2，创建名称为hhh的excel表格
    data_df.to_excel(writer, 'page_1', float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
    writer.close()
    return 1


if __name__ == '__main__':
    Algorithm1 = 'Duling-DQN'
    title1 = ''  # 读取的文件名
    title2 = ''
    title3 = ''
    # DQN
    Algorithm2 = 'DQN'
    title4 = 'E:/UAV simulation/communication_DF_db/outputs/UAV-v1/20230916-195153/results/train_ma_rewards.npy'
    title5 = 'E:/UAV simulation/communication_DF_db/outputs/UAV-v1/20230916-195153/results/train_num.npy'
    title6 = 'E:/UAV simulation/communication_DF_db/outputs/UAV-v1/20230916-195153/results/train_ma_step.npy'
    # DDQN
    Algorithm3 = 'DDQN'
    title7 = 'E:/UAV simulation/communication_DF_db/outputs/UAV-v1/20230916-195137/results/train_ma_rewards.npy'
    title8 = 'E:/UAV simulation/communication_DF_db/outputs/UAV-v1/20230916-195137/results/train_num.npy'
    title9 = 'E:/UAV simulation/communication_DF_db/outputs/UAV-v1/20230916-195137/results/train_ma_step.npy'
    # D3QN
    Algorithm4 = 'D3QN'
    title10 = 'E:/UAV simulation/communication_DF_db/outputs/UAV-v1/20230916-195145/results/train_ma_rewards.npy'
    title11 = 'E:/UAV simulation/communication_DF_db/outputs/UAV-v1/20230916-195145/results/train_num.npy'
    title12 = 'E:/UAV simulation/communication_DF_db/outputs/UAV-v1/20230916-195145/results/train_ma_step.npy'

    # improve_D3QN
    Algorithm5 = 'improve_D3QN'
    title13 = 'E:/UAV simulation/communication_DF_db/outputs/UAV-v1/20230916-195132/results/train_ma_rewards.npy'
    title14 = 'E:/UAV simulation/communication_DF_db/outputs/UAV-v1/20230916-195132/results/train_num.npy'
    title15 = 'E:/UAV simulation/communication_DF_db/outputs/UAV-v1/20230916-195132/results/train_ma_step.npy'
    # savenpyasexcel(title1, Algorithm1 + '_' + 'ma_reward')
    # savenpyasexcel(title2, Algorithm1 + '_' + 'num')
    # savenpyasexcel(title3, Algorithm1 + '_' + 'ma_step')
    savenpyasexcel(title4, Algorithm2 + '_' + 'ma_reward')
    savenpyasexcel(title5, Algorithm2 + '_' + 'num')
    savenpyasexcel(title6, Algorithm2 + '_' + 'ma_step')
    savenpyasexcel(title7, Algorithm3 + '_' + 'ma_reward')
    savenpyasexcel(title8, Algorithm3 + '_' + 'num')
    savenpyasexcel(title9, Algorithm3 + '_' + 'ma_step')
    savenpyasexcel(title10, Algorithm4 + '_' + 'ma_reward')
    savenpyasexcel(title11, Algorithm4 + '_' + 'num')
    savenpyasexcel(title12, Algorithm4 + '_' + 'ma_step')
    savenpyasexcel(title13, Algorithm5 + '_' + 'ma_reward')
    savenpyasexcel(title14, Algorithm5 + '_' + 'num')
    savenpyasexcel(title15, Algorithm5 + '_' + 'ma_step')