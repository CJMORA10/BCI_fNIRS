# bci_custom_estimators.py
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from boruta import BorutaPy


class XGBSafeBinary(BaseEstimator):
    """
    Wrapper para garantizar salida binaria consistente con labels {1,2}
    y exponer feature_importances_.
    """
    def __init__(self, **params):
        self.params = params
        self.clf_ = XGBClassifier(objective="binary:logistic", **params)
        self.le_ = LabelEncoder()

    def fit(self, X, y):
        y01 = self.le_.fit_transform(y)
        self.clf_.fit(X, y01)
        return self

    def predict(self, X):
        pred01 = self.clf_.predict(X).astype(int)
        return self.le_.inverse_transform(pred01)

    def predict_proba(self, X):
        return self.clf_.predict_proba(X)

    @property
    def classes_(self):
        return self.le_.classes_

    @property
    def feature_importances_(self):
        return self.clf_.feature_importances_

    def get_params(self, deep=True):
        return self.params.copy()

    def set_params(self, **params):
        self.params.update(params)
        self.clf_.set_params(**params)
        return self


class SafeBoruta(BaseEstimator, TransformerMixin):
    """
    Boruta que no colapsa si no selecciona nada: hace fallback
    a top-k por importancia RF.
    """
    def __init__(
        self,
        estimator,
        n_estimators="auto",
        perc=100,
        alpha=0.05,
        max_iter=100,
        random_state=None,
        verbose=0,
        fallback_frac=0.30,
        fallback_min=3,
        fallback_max=10,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.fallback_frac = fallback_frac
        self.fallback_min = fallback_min
        self.fallback_max = fallback_max
        self.boruta = None
        self.support_ = None

    def fit(self, X, y):
        self.boruta = BorutaPy(
            self.estimator,
            n_estimators=self.n_estimators,
            perc=self.perc,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.boruta.fit(X, y)

        if not any(self.boruta.support_):
            n_features_total = X.shape[1]
            n_features = int(self.fallback_frac * n_features_total)
            n_features = min(self.fallback_max, max(self.fallback_min, n_features))

            temp_rf = RandomForestClassifier(
                n_estimators=150,
                random_state=self.random_state,
                n_jobs=1,
                class_weight="balanced",
            )
            temp_rf.fit(X, y)
            importances = temp_rf.feature_importances_
            top_features = np.argsort(importances)[-n_features:]

            self.support_ = np.zeros(X.shape[1], dtype=bool)
            self.support_[top_features] = True
        else:
            self.support_ = self.boruta.support_
        return self

    def transform(self, X):
        return X[:, self.support_]

    def get_support(self, indices=False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_


class SafeSelectFromModel(BaseEstimator, TransformerMixin):
    """
    SelectFromModel con mínimo de features garantizado.
    Esto evita que XGB+SelectFromModel deje a cero columnas.
    """
    def __init__(
        self,
        estimator,
        min_features=6,
        threshold=None,
        prefit=False,
        norm_order=1,
        max_features=10,
        importance_getter="auto",
    ):
        self.estimator = estimator
        self.min_features = min_features
        self.threshold = threshold
        self.prefit = prefit
        self.norm_order = norm_order
        self.max_features = max_features
        self.importance_getter = importance_getter

        self.selector = None
        self.support_ = None

    def fit(self, X, y):
        self.selector = SelectFromModel(
            self.estimator,
            threshold=self.threshold,
            prefit=self.prefit,
            norm_order=self.norm_order,
            max_features=self.max_features,
            importance_getter=self.importance_getter,
        )
        self.selector.fit(X, y)

        support = self.selector.get_support()
        n_selected = int(support.sum())

        if n_selected < self.min_features:
            # obtiene importancias del estimator interno ya fitteado
            est = getattr(self.selector, "estimator_", None)
            if est is not None and hasattr(est, "coef_"):
                importances = np.abs(est.coef_).ravel()
            elif est is not None and hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
            else:
                importances = np.arange(X.shape[1], dtype=float)

            n_to_select = min(self.min_features, X.shape[1])
            top_idx = np.argsort(importances)[-n_to_select:]
            self.support_ = np.zeros(X.shape[1], dtype=bool)
            self.support_[top_idx] = True
        else:
            self.support_ = support

        return self

    def transform(self, X):
        return X[:, self.support_]

    def get_support(self, indices=False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_
