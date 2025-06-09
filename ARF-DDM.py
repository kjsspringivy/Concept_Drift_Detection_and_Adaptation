import pandas as pd
from layer.adaptive_learning import adaptive_learning
from layer.accuracy_plot import acc_fig
from sklearn.model_selection import train_test_split
from river import forest
from river.drift.binary import DDM

"""
Adaptive Random Forest (ARF) model with DDM drift detector (ARF-DDM)
"""

# 加载数据
df = pd.read_csv("./data/incremental_drift_data.csv")

# 训练集划分
X = df.drop(['Labelb'], axis=1)
y = df['Labelb']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.1, test_size = 0.9, shuffle=False, random_state = 0)

# Use the Adaptive Random Forest (ARF) model with DDM drift detector
name2 = "ARF-DDM model"
model2 = forest.ARFClassifier(n_models = 3, drift_detector = DDM(), seed=2025) # Define the model
t, m2, drift_points = adaptive_learning(model2, X_train, y_train, X_test, y_test) # Learn the model on the dataset
acc_fig(t, m2, name2, drift_points=drift_points) # Draw the figure of how the real-time accuracy changes with the number of samples
