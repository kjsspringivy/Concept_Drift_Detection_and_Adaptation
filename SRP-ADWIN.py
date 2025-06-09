import pandas as pd
from layer.adaptive_learning import adaptive_learning
from layer.accuracy_plot import acc_fig
from sklearn.model_selection import train_test_split
from river.drift import ADWIN
from river import ensemble


"""
Streaming Random Patches (SRP) model with ADWIN drift detector (SRP-ADWIN)
"""

# 加载数据
df = pd.read_csv("./data/incremental_drift_data.csv")

# 训练集划分
X = df.drop(['Labelb'], axis=1)
y = df['Labelb']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.2, test_size = 0.8, shuffle=False, random_state = 0)


# Use the Streaming Random Patches (SRP) model with ADWIN drift detector
name3 = "SRP-ADWIN model"
model3 = ensemble.SRPClassifier(n_models = 3, drift_detector = ADWIN(), warning_detector = ADWIN()) # Define the model
t, m3, drift_points = adaptive_learning(model3, X_train, y_train, X_test, y_test) # Learn the model on the dataset
acc_fig(t, m3, name3, drift_points=drift_points) # Draw the figure of how the real-time accuracy changes with the number of samples