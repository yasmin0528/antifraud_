import numpy as np
from sklearn.metrics import f1_score


def best_macro_f1_threshold(labels, scores):
    candidates = np.unique(np.concatenate(([0.0], np.asarray(scores), [1.0])))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        value = f1_score(labels, np.asarray(scores) >= threshold, average="macro")
        if value > best_f1:
            best_f1, best_threshold = value, float(threshold)
    return best_threshold, float(best_f1)
