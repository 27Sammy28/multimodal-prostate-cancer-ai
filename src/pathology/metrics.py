import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (y_prob >= lower) & (y_prob <= upper) if lower == 0 else (y_prob > lower) & (y_prob <= upper)
        prop = in_bin.mean()
        if prop > 0:
            ece += prop * abs(y_true[in_bin].mean() - y_prob[in_bin].mean())
    return float(ece)


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def f1_at_threshold(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def roc_curve_points(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    thresholds = y_prob[order]
    positives = max(np.sum(y_true == 1), 1)
    negatives = max(np.sum(y_true == 0), 1)
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.r_[0, tps / positives, 1]
    fpr = np.r_[0, fps / negatives, 1]
    return fpr, tpr, np.r_[np.inf, thresholds, -np.inf]


def precision_recall_points(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    thresholds = y_prob[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(np.sum(y_true == 1), 1)
    precision = np.r_[1, precision]
    recall = np.r_[0, recall]
    return precision, recall, np.r_[np.inf, thresholds]


def auc(x, y):
    order = np.argsort(x)
    x_sorted = np.asarray(x)[order]
    y_sorted = np.asarray(y)[order]
    widths = np.diff(x_sorted)
    heights = (y_sorted[:-1] + y_sorted[1:]) / 2
    return float(np.sum(widths * heights))


def calibration_curve_points(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_predicted = []
    fraction_positive = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (y_prob >= lower) & (y_prob <= upper) if lower == 0 else (y_prob > lower) & (y_prob <= upper)
        if in_bin.any():
            mean_predicted.append(y_prob[in_bin].mean())
            fraction_positive.append(y_true[in_bin].mean())
    return np.asarray(fraction_positive), np.asarray(mean_predicted)


def binary_classification_metrics(y_true, y_prob, threshold=0.5, n_bins=10):
    fpr, tpr, _ = roc_curve_points(y_true, y_prob)
    precision, recall, _ = precision_recall_points(y_true, y_prob)
    return {
        "roc_auc": auc(fpr, tpr),
        "pr_auc": auc(recall, precision),
        "f1": f1_at_threshold(y_true, y_prob, threshold),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "brier_score": brier_score(y_true, y_prob),
        "threshold": float(threshold),
    }


def curve_data(y_true, y_prob, n_bins=10):
    fpr, tpr, roc_thresholds = roc_curve_points(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_points(y_true, y_prob)
    frac_pos, mean_pred = calibration_curve_points(y_true, y_prob, n_bins=n_bins)
    return {
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "pr": {"precision": precision, "recall": recall, "thresholds": pr_thresholds},
        "calibration": {"fraction_positive": frac_pos, "mean_predicted": mean_pred},
    }