import matplotlib.pyplot as plt
import seaborn as sns


# 定义一个图形函数来显示实时精度变化
def acc_fig(t, m, name, drift_points=None):
    plt.rcParams.update({'font.size': 15})
    plt.figure(1,figsize=(10,6)) 
    sns.set_style("darkgrid")
    plt.clf() 
    plt.plot(t,m,'-b',label='Avg Accuracy: %.2f%%'%(m[-1]))
    # 标注漂移点
    if drift_points:
        for dp in drift_points:
            plt.axvline(x=dp, color='r', linestyle='--', alpha=0.7, label='Drift' if dp == drift_points[0] else "")

    plt.legend(loc='best')
    # plt.title(name+' on CICIDS2017 dataset', fontsize=15)
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy (%)')

    plt.show()