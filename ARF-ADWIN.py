import pandas as pd
from layer.adaptive_learning import adaptive_learning
from layer.accuracy_plot import acc_fig
from sklearn.model_selection import train_test_split
from river.drift import ADWIN
from river import forest


"""
Adaptive Random Forest (ARF) model with ADWIN drift detector (ARF-ADWIN)
"""

# 加载数据
df = pd.read_csv("./data/incremental_drift_data.csv")

# 训练集划分
X = df.drop(['Labelb'], axis=1)
y = df['Labelb']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.1, test_size = 0.9, shuffle=False, random_state = 0)


# 使用自适应随机森林 (ARF) 模型和 ADWIN 漂移检测器
name1 = "ARF-ADWIN model"
model1 = forest.ARFClassifier(n_models = 3, drift_detector = ADWIN(),)  # 创建一个包含3棵树的自适应随机森林，使用 ADWIN 作为漂移检测器
t, m1, drift_points = adaptive_learning(model1, X_train, y_train, X_test, y_test)  # 逐样本在线训练和测试
acc_fig(t, m1, name1, drift_points=drift_points)  # 绘制模型在测试过程中的实时准确率变化曲线
