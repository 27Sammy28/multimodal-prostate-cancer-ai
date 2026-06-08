import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


class NumpyLogisticRiskModel:
    def __init__(self, learning_rate=0.05, epochs=3000, l2=0.001):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = x.mean(axis=0)
        self.std_ = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
        x_scaled = (x - self.mean_) / self.std_
        x_design = np.c_[np.ones(len(x_scaled)), x_scaled]
        self.coef_ = np.zeros(x_design.shape[1])
        pos_weight = len(y) / max(2 * np.sum(y == 1), 1)
        neg_weight = len(y) / max(2 * np.sum(y == 0), 1)
        weights = np.where(y == 1, pos_weight, neg_weight)
        for _ in range(self.epochs):
            pred = sigmoid(x_design @ self.coef_)
            gradient = (x_design.T @ ((pred - y) * weights)) / len(y)
            gradient[1:] += self.l2 * self.coef_[1:]
            self.coef_ -= self.learning_rate * gradient
        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        x_scaled = (x - self.mean_) / self.std_
        x_design = np.c_[np.ones(len(x_scaled)), x_scaled]
        prob = sigmoid(x_design @ self.coef_)
        return np.c_[1 - prob, prob]


def build_risk_model(*_, **__):
    return NumpyLogisticRiskModel()