from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import lightgbm as lgbm
import numpy as np

class MultiModelLGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_models=10, base_estimator=lgbm.LGBMClassifier(), ):
        self.base_estimator = base_estimator
        self.n_models = n_models
        self.models = []

    def fit(self, X, y):
        split_X, split_y = np.array_split(X, self.n_models), np.array_split(y, self.n_models)
        self.models = []
        for i in range(self.n_models):
            model_clone = clone(self.base_estimator)
            model_clone.fit(split_X[i], split_y[i])
            self.models.append(model_clone)
        return self
    
    def load(self, pathname):
        for i, model in enumerate(self.models):
            self.base_estimator.Booster(model_file=f'{pathname}_{i}.txt')
        return self

    def save(self, pathname):
        for i, model in enumerate(self.models):
            model.booster_.save_model(f'{pathname}_{i}.txt')
        return self
    
    def predict_proba(self, X):
        proba_predictions = np.array([model.predict_proba(X) for model in self.models])
        mean_proba = np.mean(proba_predictions, axis=0)
        max_proba = np.max(proba_predictions, axis=0)
        min_proba = np.min(proba_predictions, axis=0)
        return mean_proba, max_proba, min_proba

