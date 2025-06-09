from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from river import metrics
from river import stream

def adaptive_learning(model, X_train, y_train, X_test, y_test):
    metric = metrics.Accuracy()
    t, m, yt, yp = [], [], [], []
    drift_points = []


    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        model.learn_one(xi1, yi1)

    for i, (xi, yi) in enumerate(stream.iter_pandas(X_test, y_test)):
        y_pred = model.predict_one(xi)

        # num_drifts_detected_last = model.n_drifts_detected()
        
        num_drifts_detected_last = sum(est.n_drifts_detected for est in model.models)
        model.learn_one(xi, yi)
        # num_drifts_detected_new = model.n_drifts_detected()
        num_drifts_detected_new = sum(est.n_drifts_detected for est in model.models)
        if num_drifts_detected_new > num_drifts_detected_last:
            drift_points.append(i)
        metric.update(yi, y_pred)
        t.append(i)
        m.append(metric.get()*100)
        yt.append(yi)
        yp.append(y_pred)

    print(f"Drift detected at sample index: {drift_points}")
    print("Accuracy: "+str(round(accuracy_score(yt, yp), 4)*100)+"%")
    print("Precision: "+str(round(precision_score(yt, yp), 4)*100)+"%")
    print("Recall: "+str(round(recall_score(yt, yp), 4)*100)+"%")
    print("F1-score: "+str(round(f1_score(yt, yp), 4)*100)+"%")
    return t, m, drift_points