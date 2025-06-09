import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb
from river import metrics
from river import stream
from layer.accuracy_plot import acc_fig

# 加载数据：前面N列为特征, 最后一列为标签
df = pd.read_csv("./data/cic_0.01km.csv")

# 训练集划分
X = df.drop(['Labelb'], axis=1)  #去掉标签列
y = df['Labelb']                 #取出标签列
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.1, test_size = 0.9, shuffle=False, random_state = 0)  # 前10%为训练集，90%为测试集

# 在线学习: LightGBM
"""
LightGBM 是一个高效的梯度提升框架, 适用于大规模数据集和高维特征, 常用于结构化数据的机器学习建模。
优势：
1. 训练速度快
2. 在结构化数据上，能捕捉复杂特征关系
3. 支持增量训练
4. 特征处理能力强：自动处理缺失值、类别特征
缺点：
1. 不是真正的在线学习模型: 主要是批量学习，每次只能用新数据重新训练或增量训练，不能像 river 等流学习库那样逐条样本实时更新
2. 对概念漂移适应性弱：模型训练后，遇到数据分布变化（概念漂移）时，不能自动适应，需要手动重训练或微调。
3. 频繁重新训练可能导致历史信息遗忘，训练成本高
"""

# 使用 LightGBM 模型
def lightGBM_learning(X_train, y_train, X_test, y_test):
    metric = metrics.Accuracy()
    i = 0
    t = []
    m = []
    yt = []
    yp = []

    model = lgb.LGBMClassifier(verbose = -1)
    model.fit(X_train,y_train)

    for xi, yi in stream.iter_pandas(X_test, y_test):
        xi2 = np.array(list(xi.values()))
        y_pred = model.predict(xi2.reshape(1, -1))      # make a prediction
        y_pred = y_pred[0]
        metric.update(yi, y_pred)  # update the metric
        
        t.append(i)
        m.append(metric.get()*100)
        yt.append(yi)
        yp.append(y_pred)
        i = i+1

    print("Accuracy: "+str(round(accuracy_score(yt,yp),4)*100)+"%")
    print("Precision: "+str(round(precision_score(yt,yp),4)*100)+"%")
    print("Recall: "+str(round(recall_score(yt,yp),4)*100)+"%")
    print("F1-score: "+str(round(f1_score(yt,yp),4)*100)+"%")
    return t, m
name0 = "LightGBM model"
t, m0 = lightGBM_learning(X_train, y_train, X_test, y_test) # Learn the LightGBM model on the dataset
acc_fig(t, m0, name0) # Draw the figure of how the real-time accuracy changes with the number of 