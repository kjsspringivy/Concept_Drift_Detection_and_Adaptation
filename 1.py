import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# # 生成1000个一维高斯噪声数据
# n_samples = 3000
# noise = np.random.normal(loc=0, scale=1, size=n_samples)

# # 可视化噪声数据
# plt.figure(figsize=(8, 5))
# plt.plot(noise, label='高斯噪声')
# plt.title('低可预测性数据', fontsize=18)
# plt.xlabel('时间', fontsize=15)
# plt.ylabel('数值', fontsize=18)
# plt.legend(fontsize=15)
# plt.tick_params(labelsize=13)
# plt.show()

datas = pd.read_csv(r"D:\频谱预测硕士\代码_练习\Transformers\datas\Spectrum5.csv", header=0, index_col=None, engine='python').values[:, 50]
# 可视化数据
plt.figure(figsize=(8, 5))
plt.plot(datas, label='频谱数据')
plt.title('频谱数据', fontsize=18)
plt.xlabel('时间', fontsize=15)
plt.ylabel('数值', fontsize=18)
plt.legend(fontsize=15)
plt.tick_params(labelsize=13)
plt.show()

