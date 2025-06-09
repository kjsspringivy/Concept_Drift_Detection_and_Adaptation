import matplotlib.pyplot as plt
from river.datasets import synth
import pandas as pd


drift_stream = synth.RandomRBFDrift(
    seed_model=42,       # RandomRBF 使用 seed_model
    seed_sample=42,      # 和 seed_sample
    n_classes=2,
    n_features=10,
    n_centroids=10,
    change_speed=0.01  # RandomRBF 使用 change_speed
)
# 2. 生成突变漂移数据流
total_samples = 3000
# drift_position = 1500  # 在第1500个样本发生漂移
# drift_width = 10      
# drift_stream = synth.ConceptDriftStream(
#     seed=42,
#     position=drift_position,
#     width=40
# )

x_vals, y_vals = [], []
for x, y in drift_stream.take(total_samples):
    x_vals.append(list(x.values()))
    y_vals.append(y)

# 3. 可视化漂移效果
# 我们通过绘制数据流的“早期”和“晚期”快照来对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
fig.suptitle('Dataset with Incremental Drift (RBF Generator)', fontsize=16)

# 提取特征用于绘图
feature0 = [x[0] for x in x_vals]
feature1 = [x[1] for x in x_vals]
# feature2 = [x[2] for x in x_vals]  # 如果需要，可以添加更多特征
# feature3 = [x[3] for x in x_vals]  # 如果需要，可以添加更多特征

# --- 绘制第一个快照: 数据流的早期 (前1000个样本) ---
num_snapshot_samples = 1000
scatter1 = ax1.scatter(
    feature0[:num_snapshot_samples],
    feature1[:num_snapshot_samples],
    c=y_vals[:num_snapshot_samples],
    cmap='viridis',
    alpha=0.7,
    s=15
)
ax1.set_title(f'Data Distribution (First {num_snapshot_samples} Instances)')
ax1.set_xlabel('Feature 0')
ax1.set_ylabel('Feature 1')
ax1.grid(True, linestyle='--', alpha=0.6)
# 修正图例以匹配 n_classes
# legend_elements() 会自动从数据中获取类别
handles1, auto_labels1 = scatter1.legend_elements()
# 如果需要自定义标签文本：
class_labels = [f'Class {i}' for i in range(drift_stream.n_classes)]
ax1.legend(handles=handles1, labels=class_labels[:len(handles1)])

# --- 绘制第二个快照: 数据流的晚期 (后1000个样本) ---
scatter2 = ax2.scatter(
    feature0[-num_snapshot_samples:],
    feature1[-num_snapshot_samples:],
    c=y_vals[-num_snapshot_samples:],
    cmap='viridis',
    alpha=0.7,
    s=15
)
ax2.set_title(f'Data Distribution (Last {num_snapshot_samples} Instances)')
ax2.set_xlabel('Feature 0')
ax2.grid(True, linestyle='--', alpha=0.6)
# 修正图例
handles2, auto_labels2 = scatter2.legend_elements()
ax2.legend(handles=handles2, labels=class_labels[:len(handles2)])

plt.tight_layout() # 调整布局以适应标题
plt.show()

# 4. 保存数据到 CSV 文件
data = pd.DataFrame(x_vals, columns=[f'Feature_{i}' for i in range(len(x_vals[0]))])
data['Labelb'] = y_vals
data.to_csv(r'D:/频谱预测硕士/代码_练习/Concept_Drift_Detection_and_Adaptation/data/incremental_drift_data.csv', index=False)


