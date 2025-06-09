import pandas as pd
from layer.adaptive_learning import adaptive_learning
from layer.accuracy_plot import acc_fig
from sklearn.model_selection import train_test_split
from river import tree

"""
Hoeffding Tree (HT) 模型
"""
# 加载数据
df = pd.read_csv("./data/cic_0.01km.csv")

# 训练集划分
X = df.drop(['Labelb'], axis=1)
y = df['Labelb']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.1, test_size = 0.9, shuffle=False, random_state = 0)

# 使用 Hoeffding Tree (HT) 模型
name6 = "HT model"
model6 = tree.HoeffdingTreeClassifier() # Define the model
t, m6 = adaptive_learning(model6, X_train, y_train, X_test, y_test) # Learn the model on the dataset
acc_fig(t, m6, name6) # Draw the figure of how the real-time accuracy changes with the number of samples