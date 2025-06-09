import matplotlib.pyplot as plt
from river import stream, drift, ensemble, tree
from river.datasets import synth
from river.drift.binary import DDM

# 1. 生成概念漂移数据集
# 创建一个数据流，其中概念在样本1000处发生变化
stream_generator = synth.ConceptDriftStream(
    stream=synth.Hyperplane(seed=42, n_features=2),
    drift_stream=synth.Hyperplane(seed=42, n_features=2, mag_change=0.5),
    position=1000,
    width=50,
    seed=1
)

# --- DDM 漂移检测 ---
print("Running DDM (Drift Detection Method)...")
ddm = ()
ht = tree.HoeffdingTreeClassifier()
ddm_metrics = {'n_samples': [], 'accuracy': [], 'drift_points': []}
correct_predictions_ddm = 0

# 迭代处理数据流
for i, (X, y) in enumerate(stream_generator.take(2000)):
    prediction = ht.predict_one(X)
    is_correct = (prediction == y)
    if is_correct:
        correct_predictions_ddm += 1

    # DDM 检查错误率
    ddm.update(int(not is_correct))
    if ddm.drift_detected:
        print(f"DDM: Drift detected at sample {i}")
        ddm_metrics['drift_points'].append(i)
        # 漂移发生后，重置模型
        ht = tree.HoeffdingTreeClassifier()

    ht.learn_one(X, y)

    # 记录数据用于绘图
    ddm_metrics['n_samples'].append(i)
    ddm_metrics['accuracy'].append(correct_predictions_ddm / (i + 1))

print("-" * 30)

# --- ARF 漂移检测与适应 ---
print("\nRunning ARF (Adaptive Random Forest)...")
# ARF 内置了漂移检测和适应机制
arf = ensemble.AdaptiveRandomForestClassifier(seed=42, n_models=3)
arf_metrics = {'n_samples': [], 'accuracy': [], 'drift_points': []}
correct_predictions_arf = 0

# 重置数据流生成器
stream_generator.restart()

# 迭代处理数据流
for i, (X, y) in enumerate(stream_generator.take(2000)):
    prediction = arf.predict_one(X)
    is_correct = (prediction == y)
    if is_correct:
        correct_predictions_arf += 1

    arf.learn_one(X, y)

    # ARF内部会处理漂移，我们这里只是为了演示而记录
    # 在实际应用中，ARF会自动替换表现不佳的树
    # 为了可视化，我们可以检查每个内部树的漂移检测器状态
    for tree_idx, tree_model in enumerate(arf):
        if tree_model.drift_detector.drift_detected:
             if i not in arf_metrics['drift_points']: # 避免重复记录
                print(f"ARF: Drift detected in tree {tree_idx} at sample {i}")
                arf_metrics['drift_points'].append(i)


    # 记录数据用于绘图
    arf_metrics['n_samples'].append(i)
    arf_metrics['accuracy'].append(correct_predictions_arf / (i + 1))

print("Done.")


# --- 结果可视化 ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# 绘制 DDM 结果
ax1.plot(ddm_metrics['n_samples'], ddm_metrics['accuracy'], label='Hoeffding Tree with DDM', color='dodgerblue')
ax1.axvline(x=1000, color='grey', linestyle='--', label='Concept Drift Point (Sample 1000)')
for drift_point in ddm_metrics['drift_points']:
    ax1.axvline(x=drift_point, color='red', linestyle='-.', label=f'DDM Detected Drift ({drift_point})')
ax1.set_title('DDM Drift Detection', fontsize=14)
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# 绘制 ARF 结果
ax2.plot(arf_metrics['n_samples'], arf_metrics['accuracy'], label='Adaptive Random Forest (ARF)', color='forestgreen')
ax2.axvline(x=1000, color='grey', linestyle='--', label='Concept Drift Point (Sample 1000)')
for drift_point in arf_metrics['drift_points']:
    ax2.axvline(x=drift_point, color='red', linestyle='-.', label=f'ARF Detected Drift ({drift_point})')
ax2.set_title('ARF Drift Detection and Adaptation', fontsize=14)
ax2.set_xlabel('Samples')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()