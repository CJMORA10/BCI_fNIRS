import numpy as np
import scipy.io as sio
import pickle
import pandas as pd
from pathlib import Path

# ==== imports necesarios para las clases personalizadas ====
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from boruta import BorutaPy

# =============================================================================
#   CLASES PERSONALIZADAS PARA SELECCIÓN DE CARACTERÍSTICAS
# =============================================================================

MIN_FEAT = 6
MAX_FEAT = 10

class XGBSafeBinary(BaseEstimator):
    def __init__(self, **params):
        self.params = params
        self.clf_ = XGBClassifier(objective='binary:logistic', **params)
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
    def __init__(self, estimator, n_estimators='auto', perc=100, alpha=0.05,
                 max_iter=100, random_state=None, verbose=0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.boruta = None

    def fit(self, X, y):
        self.boruta = BorutaPy(
            self.estimator,
            n_estimators=self.n_estimators,
            perc=self.perc,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.boruta.fit(X, y)

        if not any(self.boruta.support_):
            n_features_total = X.shape[1]
            n_features = min(10, max(3, int(0.3 * n_features_total)))
            temp_rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=1,
                class_weight='balanced'
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
    def __init__(self, estimator, min_features=MIN_FEAT, threshold=None,
                 prefit=False, norm_order=1, max_features=MAX_FEAT,
                 importance_getter='auto'):
        self.estimator = estimator
        self.min_features = min_features
        self.threshold = threshold
        self.prefit = prefit
        self.norm_order = norm_order
        self.max_features = max_features
        self.importance_getter = importance_getter
        self.selector = None

    def fit(self, X, y):
        self.selector = SelectFromModel(
            self.estimator,
            threshold=self.threshold,
            prefit=self.prefit,
            norm_order=self.norm_order,
            max_features=self.max_features,
            importance_getter=self.importance_getter
        )
        self.selector.fit(X, y)
        support = self.selector.get_support()
        n_selected = support.sum()

        if n_selected < self.min_features:
            if hasattr(self.selector.estimator_, 'coef_'):
                importances = np.abs(self.selector.estimator_.coef_).ravel()
            elif hasattr(self.selector.estimator_, 'feature_importances_'):
                importances = self.selector.estimator_.feature_importances_
            else:
                importances = np.arange(X.shape[1])
            n_to_select = min(self.min_features, X.shape[1])
            top_indices = np.argsort(importances)[-n_to_select:]
            self.support_ = np.zeros(X.shape[1], dtype=bool)
            self.support_[top_indices] = True
        else:
            self.support_ = support
        return self

    def transform(self, X):
        return X[:, self.support_]

    def get_support(self, indices=False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_


# =============================================================================
#   RESTO DEL SCRIPT (exportar comandos para los 28 sujetos)
# =============================================================================

FEAT_PER_CH = 7

def slope(x):
    return np.diff(x, axis=1).mean(1)

def features(O, R, t_mask, ch):
    O_ = O[:, t_mask][:, :, ch]
    R_ = R[:, t_mask][:, :, ch]
    f = [
        O_.mean(1),
        slope(O_),
        R_.mean(1),
        slope(R_),
        O_.max(1),
        O_.var(1),
        (O_ - R_).mean(1)
    ]
    F = np.stack(f, axis=2)
    return F.reshape(O_.shape[0], -1)

def cols_from_channels(ch_sel):
    return np.concatenate([
        np.arange(c * FEAT_PER_CH, (c + 1) * FEAT_PER_CH) for c in ch_sel
    ])

def probs_to_logit_pairs(Z, eps=1e-6):
    Z = np.clip(Z, eps, 1 - eps)
    exps = Z.shape[1] // 2
    feats = []
    for i in range(exps):
        p1 = Z[:, 2 * i]
        p2 = Z[:, 2 * i + 1]
        feats.append(np.log(p2 / p1))
    return np.column_stack(feats)

BASE = Path("TRAINED_MODEL_IMPROVED_V5_SELECTIVE")
OUT = Path("Comandos_V5")
SUBJECTS = [f"subject {i:02}" for i in range(1, 29)]

label_name = {0: "idle", 1: "right", 2: "left"}

def export_subject_commands(subj_name: str):
    subj_dir = BASE / subj_name
    out_dir = OUT / subj_name
    out_dir.mkdir(parents=True, exist_ok=True) 

    if not subj_dir.exists():
        print(f"[{subj_name}] carpeta no encontrada, se omite.")
        return

    model_path   = subj_dir / f"{subj_name}_model_ensemble.pkl"
    holdout_path = subj_dir / f"{subj_name}_holdout25.mat"

    if not model_path.exists() or not holdout_path.exists():
        print(f"[{subj_name}] faltan archivos .pkl o .mat, se omite.")
        return

    print(f"=== Exportando comandos para {subj_name} ===")

    # cargar modelo
    with open(model_path, "rb") as f:
        model_pkg = pickle.load(f)

    # cargar datos hold-out
    m = sio.loadmat(holdout_path, squeeze_me=True)
    O = m["O"]
    R = m["R"]
    y_true = m["labels"].astype(int).ravel()

    # info del modelo
    t_mask     = model_pkg["t_mask"]
    ch_sel     = model_pkg["channels"]
    thr_reject = model_pkg["thr_reject"]
    models     = model_pkg["models"]
    meta       = model_pkg["stacking_meta"]

    # features como en entrenamiento
    cols_final = cols_from_channels(ch_sel)
    X_all      = features(O, R, t_mask, np.arange(24)).astype(np.float32, copy=False)
    X_feat     = X_all[:, cols_final]

    # probs de cada experto
    Z_list = []
    for name, pipe in models:
        Z_list.append(pipe.predict_proba(X_feat))
    Z_full = np.concatenate(Z_list, axis=1)
    Z_meta = probs_to_logit_pairs(Z_full)

    # meta + rechazo
    proba  = meta.predict_proba(Z_meta)
    conf   = proba.max(1)
    y_pred = proba.argmax(1) + 1
    y_pred[conf < thr_reject] = 0

    # dataframe de salida
    df_out = pd.DataFrame({
        "trial":      np.arange(len(y_pred)),
        "true_label": y_true,
        "pred_label": y_pred,
    })
    df_out["pred_name"] = df_out["pred_label"].map(label_name)

    num = int(subj_name.split()[1])
    out_csv = out_dir / f"commands_subject{num:02}.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"[{subj_name}] CSV guardado en: {out_csv}")

if __name__ == "__main__":
    for subj in SUBJECTS:
        export_subject_commands(subj)

    print("\nExportación de comandos completada para todos los sujetos.")
