import lightgbm as lgb
import numpy as np
from scipy.special import expit

class GroupDROObjective:

    def __init__(self, group_labels, step_size=0.01, scale_pos_weight=3):
        self.group_labels = np.array(group_labels)
        self.step_size = step_size
        self.scale_pos_weight = scale_pos_weight
        self.unique_groups = np.unique(self.group_labels)
        self.n_groups = len(self.unique_groups)
        self.group_weights = np.ones(self.n_groups) / self.n_groups

    # sklearn API objective signature: (y_true, y_pred) -> (grad, hess)
    # y_pred are raw scores (log-odds), not probabilities.
    def __call__(self, y_true, y_pred):
        probs = expit(y_pred)
        eps = 1e-15
        sample_losses = -(y_true * np.log(probs + eps) + (1 - y_true) * np.log(1 - probs + eps))
        group_losses = np.zeros(self.n_groups)
        for idx, g in enumerate(self.unique_groups):
            mask = self.group_labels == g
            if np.any(mask):
                group_losses[idx] = np.mean(sample_losses[mask])
        self.group_weights *= np.exp(self.step_size * group_losses)
        self.group_weights /= np.sum(self.group_weights)
        sample_weights = np.zeros(len(y_true))
        for idx, g in enumerate(self.unique_groups):
            mask = self.group_labels == g
            group_size_ratio = np.sum(mask) / len(y_true)
            sample_weights[mask] = self.group_weights[idx] / (group_size_ratio + eps)
        # Class imbalance: scale_pos_weight not applied when using custom objective,
        # so replicate it here by upweighting positive samples.
        sample_weights = np.where(y_true == 1, sample_weights * self.scale_pos_weight, sample_weights)
        grad = (probs - y_true) * sample_weights
        hess = probs * (1 - probs) * sample_weights
        return grad, hess

class ModelGBT:

    def __init__(self, n_estimators, learning_rate, max_depth, scale_pos_weight, random_state=42):
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._scale_pos_weight = scale_pos_weight
        self._random_state = random_state
        self._classifier = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective='binary',
            scale_pos_weight=scale_pos_weight,
            force_row_wise=True,
            n_jobs=-1,
            random_state=random_state,
        )
        self._dro_classifier = None

    def fit(self, X, y):
        self._dro_classifier = None
        self._classifier.fit(X, y)

    def fit_group_dro(self, X, y, group_labels, step_size=0.01):
        dro_obj = GroupDROObjective(
            group_labels,
            step_size=step_size,
            scale_pos_weight=self._scale_pos_weight,
        )
        self._dro_classifier = lgb.LGBMClassifier(
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            max_depth=self._max_depth,
            objective=dro_obj,
            force_row_wise=True,
            n_jobs=-1,
            random_state=self._random_state,
        )
        self._dro_classifier.fit(X, y)

    def predict(self, X):
        if self._dro_classifier is not None:
            raw = self._dro_classifier.booster_.predict(X)
            return expit(raw)
        return self._classifier.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self):
        if self._dro_classifier is not None:
            return self._dro_classifier.booster_.feature_importance(importance_type='split')
        return self._classifier.feature_importances_