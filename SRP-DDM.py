import pandas as pd
from layer.adaptive_learning import adaptive_learning
from layer.accuracy_plot import acc_fig
from sklearn.model_selection import train_test_split
from river.drift.binary import DDM
from river import ensemble


"""
Streaming Random Patches (SRP) model with DDM drift detector (SRP-DDM)
"""

# 加载数据
df = pd.read_csv("./data/incremental_drift_data.csv")

# 训练集划分
X = df.drop(['Labelb'], axis=1)
y = df['Labelb']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.1, test_size = 0.9, shuffle=False, random_state = 0)


# 使用带有 DDM 漂移检测器的流随机补丁 (SRP) 模型
name4 = "SRP-DDM model"
model4 = ensemble.SRPClassifier(n_models = 3, drift_detector = DDM(), warning_detector=DDM()) # Define the model
t, m4, drift_points = adaptive_learning(model4, X_train, y_train, X_test, y_test) # Learn the model on the dataset
acc_fig(t, m4, name4, drift_points=drift_points) # Draw the figure of how the real-time accuracy changes with the number of samples
