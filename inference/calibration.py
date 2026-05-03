import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration

class PlattCalibrator:

    def __init__(self):
        self._model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self._fitted = False

    def fit(self, logits, labels):
        self._model.fit(logits.reshape(-1, 1), labels)
        self._fitted = True

    def transform(self, logits):
        if not self._fitted:
            raise RuntimeError("PlattCalibrator has not been fitted yet.")
        return self._model.predict_proba(logits.reshape(-1, 1))[:, 1]

class IsotonicCalibrator:

    def __init__(self):
        self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        self._fitted = False

    def fit(self, predictions, labels):
        self._model.fit(predictions, labels)
        self._fitted = True

    def transform(self, predictions):
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator has not been fitted yet.")
        return self._model.transform(predictions)

# Beta calibration fits a smooth 3-parameter map from raw scores to probabilities.
class BetaCalibrator:

    def __init__(self):
        self._model = BetaCalibration(parameters='abm')
        self._fitted = False

    def fit(self, predictions, labels):
        self._model.fit(predictions.reshape(-1, 1), labels)
        self._fitted = True

    def transform(self, predictions):
        if not self._fitted:
            raise RuntimeError("BetaCalibrator has not been fitted yet.")
        return self._model.predict(predictions.reshape(-1, 1))

class NNBetaCalibrator:

    def __init__(self):
        self._model = BetaCalibration(parameters='abm')
        self._fitted = False

    def fit(self, logits, labels):
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        self._model.fit(probs.reshape(-1, 1), labels)
        self._fitted = True

    def transform(self, logits):
        if not self._fitted:
            raise RuntimeError("NNBetaCalibrator has not been fitted yet.")
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        return self._model.predict(probs.reshape(-1, 1))

def find_threshold(predictions, labels, target_precision, min_predictions=10, fallback_percentile=99):
    best_threshold = None
    for t in np.arange(0.01, 1.00, 0.01):
        predicted_positive = predictions >= t
        n_positive = predicted_positive.sum()
        if n_positive < min_predictions:
            continue
        true_positive = predicted_positive & (labels == 1.0)
        precision = true_positive.sum() / n_positive
        if precision >= target_precision:
            best_threshold = t
            break
    if best_threshold is None:
        best_threshold = float(np.percentile(predictions, fallback_percentile))
    return best_threshold