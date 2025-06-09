import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from river import metrics
from river import stream
from river import ensemble
from river.drift.binary import DDM
from river.drift import ADWIN
from river import forest
from layer.accuracy_plot import acc_fig

# Define the Performance Weighted Probability Averaging Ensemble (PWPAE) model
def PWPAE(X_train, y_train, X_test, y_test):
    # Record the real-time accuracy of PWPAE and 4 base learners
    metric = metrics.Accuracy()
    metric1 = metrics.Accuracy()
    metric2 = metrics.Accuracy()
    metric3 = metrics.Accuracy()
    metric4 = metrics.Accuracy()

    i=0
    t = []
    m = []
    yt = []
    yp = []

    hat1 = forest.ARFClassifier(n_models=3, drift_detector = ADWIN()) # ARF-ADWIN
    hat2 = ensemble.SRPClassifier(n_models=3, drift_detector = ADWIN()) # SRP-ADWIN
    hat3 = forest.ARFClassifier(n_models=3,drift_detector=DDM(),warning_detector=DDM()) # ARF-DDM
    hat4 = ensemble.SRPClassifier(n_models=3,drift_detector=DDM(),warning_detector=DDM()) # SRP-DDM

    # 基分类器的训练
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        hat1.learn_one(xi1, yi1)
        hat2.learn_one(xi1, yi1)
        hat3.learn_one(xi1, yi1)
        hat4.learn_one(xi1, yi1)

    # 预测
    for xi, yi in stream.iter_pandas(X_test, y_test):
        # The four base learner predict the labels
        y_pred1= hat1.predict_one(xi) 
        y_prob1= hat1.predict_proba_one(xi)  # {0: 0.3, 1: 0.7}
        hat1.learn_one(xi, yi)

        y_pred2= hat2.predict_one(xi) 
        y_prob2= hat2.predict_proba_one(xi)
        hat2.learn_one(xi, yi)

        y_pred3= hat3.predict_one(xi) 
        y_prob3= hat3.predict_proba_one(xi)
        hat3.learn_one(xi, yi)

        y_pred4= hat4.predict_one(xi) 
        y_prob4= hat4.predict_proba_one(xi)
        hat4.learn_one(xi, yi)
        
        # Record their real-time accuracy
        metric1.update(yi, y_pred1)
        metric2.update(yi, y_pred2)
        metric3.update(yi, y_pred3)
        metric4.update(yi, y_pred4)    

        # Calculate the real-time error rates of four base learners
        e1 = 1-metric1.get()
        e2 = 1-metric2.get()
        e3 = 1-metric3.get()
        e4 = 1-metric4.get()

        
        ep = 0.001  # 防止除零
        # 根据错误率赋值权重
        ea = 1/(e1+ep)+1/(e2+ep)+1/(e3+ep)+1/(e4+ep)
        w1 = 1/(e1+ep)/ea
        w2 = 1/(e2+ep)/ea
        w3 = 1/(e3+ep)/ea
        w4 = 1/(e4+ep)/ea

        # 每个基分类器属于0类或1类的概率
        if  y_pred1 == 1:
            ypro10 = 1-y_prob1[1]  # 属于0类的概率
            ypro11 = y_prob1[1]  # 属于1类的概率
        else:
            ypro10 = y_prob1[0]
            ypro11=1-y_prob1[0]

        if  y_pred2 == 1:
            ypro20 = 1-y_prob2[1]
            ypro21 = y_prob2[1]
        else:
            ypro20 = y_prob2[0]
            ypro21 = 1-y_prob2[0]

        if  y_pred3 == 1:
            ypro30 = 1-y_prob3[1]
            ypro31 = y_prob3[1]
        else:
            ypro30 = y_prob3[0]
            ypro31 = 1-y_prob3[0]

        if  y_pred4 == 1:
            ypro40 = 1-y_prob4[1]
            ypro41 = y_prob4[1]
        else:
            ypro40 = y_prob4[0]
            ypro41 = 1-y_prob4[0]        

        # 计算 0 类和 1 类的最终概率以进行预测
        y_prob_0 = w1*ypro10+w2*ypro20+w3*ypro30+w4*ypro40  # 属于0类的概率
        y_prob_1 = w1*ypro11+w2*ypro21+w3*ypro31+w4*ypro41  # 属于1类的概率

        if (y_prob_0 > y_prob_1):
            y_pred = 0
            # y_prob = y_prob_0
        else:
            y_pred = 1
            # y_prob = y_prob_1
        
        # 更新准确率
        metric.update(yi, y_pred)

        t.append(i)
        m.append(metric.get()*100)
        yt.append(yi)
        yp.append(y_pred)
        i=i+1

    print("Accuracy: "+str(round(accuracy_score(yt,yp),4)*100)+"%")
    print("Precision: "+str(round(precision_score(yt,yp),4)*100)+"%")
    print("Recall: "+str(round(recall_score(yt,yp),4)*100)+"%")
    print("F1-score: "+str(round(f1_score(yt,yp),4)*100)+"%")
    return t, m

# 加载数据
df = pd.read_csv("./data/cic_0.01km.csv")

# 训练集划分
X = df.drop(['Labelb'], axis=1)
y = df['Labelb']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.1, test_size = 0.9, shuffle=False, random_state = 0)

name = "Proposed PWPAE model"
t, m = PWPAE(X_train, y_train, X_test, y_test) # Learn the model on the dataset
acc_fig(t, m, name) # Draw the figure of how the real-time accuracy changes with the number of samples


