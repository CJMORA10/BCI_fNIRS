# -*- coding: utf-8 -*-
"""
BCI-fNIRS · Reporte(stacking + rechazo) - PLOTS por sujeto y global

Estructura esperada:
TRAINED_MODEL_IMPROVED_V4_FINAL/
  subject 01/
    subject 01_model_ensemble.pkl
    subject 01_holdout25.mat
  subject 02/
    ...

Salida:
BCI_REPORT_V4/
  subject 01/ (figuras + csv)
  ...
  GLOBAL/    (figuras globales + csv resumen)
"""

import os
import sys
import types
import pickle
import warnings
from pathlib import Path
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import scipy.io as sio
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

# =========================================================
# 0) PLACEHOLDERS / CLASES EXACTAS (para unpickle correcto)
#    (Esto evita: "Can't get attribute 'SafeSelectFromModel' on __main__")
# =========================================================

# xgboost puede o no estar instalado
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# boruta puede o no estar instalado
try:
    from boruta import BorutaPy
except Exception:
    BorutaPy = None


def _install_dummy_boruta_module():
    """
    Si no está boruta instalado pero el pickle lo referencia, creamos módulos dummy
    para que pickle.load no falle.
    """
    if BorutaPy is not None:
        return

    boruta_mod = types.ModuleType("boruta")
    boruta_py_mod = types.ModuleType("boruta.boruta_py")

    class BorutaPyDummy(BaseEstimator, TransformerMixin):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.support_ = None

        def fit(self, X, y=None):
            self.support_ = np.ones(X.shape[1], dtype=bool)
            return self

        def transform(self, X):
            return X

        def get_support(self, indices=False):
            if indices:
                return np.arange(X.shape[1])
            return np.ones(X.shape[1], dtype=bool)

    boruta_mod.BorutaPy = BorutaPyDummy
    boruta_py_mod.BorutaPy = BorutaPyDummy

    sys.modules["boruta"] = boruta_mod
    sys.modules["boruta.boruta_py"] = boruta_py_mod


# === Clases como en tu entrenamiento (mismo nombre y API) ===

class XGBSafeBinary(BaseEstimator):
    """
    Wrapper usado en el entrenamiento.
    Importante para que el estado (clf_ y le_) se recupere bien al cargar el pickle.
    """
    def __init__(self, **params):
        if XGBClassifier is None:
            raise ImportError("xgboost no está disponible pero el modelo lo requiere.")
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
    """
    Igual al entrenamiento (usa BorutaPy si existe; si no, deja pasar).
    """
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
        self.support_ = None

    def fit(self, X, y):
        if BorutaPy is None:
            # fallback: no seleccionar nada, dejar todo
            self.support_ = np.ones(X.shape[1], dtype=bool)
            return self

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
            # fallback como en el entrenamiento
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
        return X[:, self.support_] if self.support_ is not None else X

    def get_support(self, indices=False):
        if self.support_ is None:
            self.support_ = np.ones(1, dtype=bool)
        if indices:
            return np.where(self.support_)[0]
        return self.support_


class SafeSelectFromModel(BaseEstimator, TransformerMixin):
    """
    Igual a tu entrenamiento (NO es el SelectFromModel de sklearn directamente).
    """
    def __init__(self, estimator, min_features=6, threshold=None, prefit=False,
                 norm_order=1, max_features=10, importance_getter='auto'):
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
            importance_getter=self.importance_getter
        )
        self.selector.fit(X, y)
        support = self.selector.get_support()
        n_selected = support.sum()

        if n_selected < self.min_features:
            # fallback como en tu entrenamiento
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
        return X[:, self.support_] if self.support_ is not None else X

    def get_support(self, indices=False):
        if self.support_ is None:
            self.support_ = np.ones(1, dtype=bool)
        if indices:
            return np.where(self.support_)[0]
        return self.support_


# =========================================================
# 1) Config
# =========================================================

BASE = Path("TRAINED_MODEL_IMPROVED_V5_SELECTIVE")
OUT  = Path("BCI_REPORT_V5")
OUT.mkdir(parents=True, exist_ok=True)
SUBJECTS = [f"subject {i:02d}" for i in range(1, 29)]
LABELS_3 = [1, 2, 0]  # Mano derecha, Mano izquierda, Rechazo
NAMES_3  = ["Mano derecha", "Mano izquierda", "Rechazo"]

LABELS_2 = [1, 2]
NAMES_2  = ["Mano derecha", "Mano izquierda"]

FEAT_PER_CH = 7


# =========================================================
# 2) Features (idéntico orden al entrenamiento)
# =========================================================

def slope(x):
    return np.diff(x, axis=1).mean(1)

def features(O, R, t_mask, ch_idx):
    """
    Replica exacta del entrenamiento:
    O_, R_ => [N, Tmask, C]
    features por canal: [O_mean, O_slope, R_mean, R_slope, O_peak, O_var, (O-R)_mean]
    salida: [N, C*7]
    """
    O_ = O[:, t_mask][:, :, ch_idx]
    R_ = R[:, t_mask][:, :, ch_idx]
    f = [
        O_.mean(1),
        slope(O_),
        R_.mean(1),
        slope(R_),
        O_.max(1),
        O_.var(1),
        (O_ - R_).mean(1),
    ]
    F = np.stack(f, axis=2)              # [N, C, 7]
    return F.reshape(O_.shape[0], -1)    # [N, C*7]


def balanced_accuracy_safe(y_true, y_pred, labels=(1, 2)):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    for c in labels:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = tp + fn
        recalls.append(0.5 if denom == 0 else tp / denom)
    return float(np.mean(recalls))


def probs_to_logit_pairs(Z, eps=1e-6):
    """
    Igual al entrenamiento:
    Z tiene 2 columnas por experto (p1,p2) -> devuelve 1 col por experto: log(p2/p1)
    """
    Z = np.clip(Z, eps, 1 - eps)
    exps = Z.shape[1] // 2
    feats = []
    for i in range(exps):
        p1 = Z[:, 2 * i]
        p2 = Z[:, 2 * i + 1]
        feats.append(np.log(p2 / p1))
    return np.column_stack(feats)


# =========================================================
# 3) Load ensemble y predicción (idéntico concepto al entrenamiento)
# =========================================================

def load_ensemble(pkl_path: Path):
    _install_dummy_boruta_module()
    with open(pkl_path, "rb") as f:
        ens = pickle.load(f)
    return ens


def _proba_as_p1p2(model, X):
    """
    Devuelve proba en orden [p(clase1), p(clase2)] aunque classes_ esté invertido.
    """
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    if proba.shape[1] != 2:
        raise ValueError("Se esperaba proba binaria (2 columnas).")

    i1 = int(np.where(classes == 1)[0][0])
    i2 = int(np.where(classes == 2)[0][0])
    return np.column_stack([proba[:, i1], proba[:, i2]])


def predict_meta_with_rejection(ens, X, y_true):
    """
    1) Z_full = concat de predict_proba de expertos (2 cols por experto)
    2) decide si meta espera logit-pairs o raw según n_features_in_
    3) proba_final = meta.predict_proba(...)
    4) rechazo si conf < thr_used
    """
    experts = OrderedDict((name, pipe) for name, pipe in ens["models"])
    expert_names = ens.get("expert_names", list(experts.keys()))
    meta = ens.get("stacking_meta", None)

    if meta is None:
        raise RuntimeError("No se encontró stacking_meta en el pkl.")

    # --- construir Z_full respetando el orden de expert_names
    Z_list = []
    ok = []
    for nm in expert_names:
        if nm not in experts:
            continue
        mdl = experts[nm]
        try:
            Z_list.append(_proba_as_p1p2(mdl, X))
            ok.append(nm)
        except Exception as e:
            warnings.warn(f"[{ens.get('win_key','?')}] Experto '{nm}' falló predict_proba: {e}")

    if len(Z_list) == 0:
        raise RuntimeError("Ningún experto pudo predecir.")

    Z_full = np.concatenate(Z_list, axis=1)  # [N, 2*n_exp_ok]

    # --- decidir representación de meta: logit o raw
    p_expected = getattr(meta, "n_features_in_", None)
    if p_expected is None:
        # fallback: intentar raw y si falla, logit
        try:
            Z_meta = Z_full
            proba_final = meta.predict_proba(Z_meta)
        except Exception:
            Z_meta = probs_to_logit_pairs(Z_full)
            proba_final = meta.predict_proba(Z_meta)
    else:
        if p_expected == Z_full.shape[1]:
            Z_meta = Z_full
        elif p_expected == (Z_full.shape[1] // 2):
            Z_meta = probs_to_logit_pairs(Z_full)
        else:
            # último intento: si coincide con #expertos ok
            if p_expected == len(ok):
                Z_meta = probs_to_logit_pairs(Z_full)
            else:
                raise RuntimeError(f"Meta espera {p_expected} features, pero tengo {Z_full.shape[1]} (raw) / {Z_full.shape[1]//2} (logit).")

        proba_final = meta.predict_proba(Z_meta)

    # Alinear columnas de meta a clase 1 y 2
    mclasses = np.asarray(meta.classes_)
    i1 = int(np.where(mclasses == 1)[0][0])
    i2 = int(np.where(mclasses == 2)[0][0])
    proba_12 = np.column_stack([proba_final[:, i1], proba_final[:, i2]])

    # Predicción sin rechazo (forced)
    y_pred_forced = np.where(proba_12[:, 1] >= 0.5, 2, 1)

    # Rechazo
    thr = float(ens.get("thr_reject", 0.0))
    target_rej = ens.get("target_rej", None)

    conf = proba_12.max(1)
    # "recentering" suave como el entrenamiento
    if target_rej is not None:
        cur_rej = float(np.mean(conf < thr))
        if abs(cur_rej - float(target_rej)) > 0.08:
            q = float(np.clip(target_rej, 0.01, 0.99))
            thr = float(np.quantile(conf, q))

    y_pred_rej = y_pred_forced.copy()
    y_pred_rej[conf < thr] = 0

    return dict(
        proba_12=proba_12,
        conf=conf,
        thr_used=thr,
        y_pred_forced=y_pred_forced,
        y_pred_rej=y_pred_rej,
        experts_ok=ok,
        Z_full=Z_full
    )


# =========================================================
# 4) Métricas extra: ECE y plots
# =========================================================

def expected_calibration_error(y_true_bin, p_pos, n_bins=10):
    y_true_bin = np.asarray(y_true_bin).astype(int)
    p_pos = np.asarray(p_pos).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p_pos >= lo) & (p_pos < hi) if i < n_bins-1 else (p_pos >= lo) & (p_pos <= hi)
        if not np.any(mask):
            continue
        acc = y_true_bin[mask].mean()
        conf = p_pos[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def plot_confusion(cm, class_names, title, out_png):
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=160)
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, str(val),
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=13)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_acc_and_calibration(y_true, proba_12, thr_used, out_png,
                                      title, n_bins=10,
                                      thr_min=0.50, thr_max=0.95, n_thr=61):
    """
    Subplot 1: Accuracy(aceptados) vs Rejection rate (barrido de umbral)
    Subplot 2: Calibration curve (reliability) + ECE
    """
    y_true = np.asarray(y_true).astype(int)
    proba_12 = np.asarray(proba_12, dtype=float)

    p2 = proba_12[:, 1]
    conf = proba_12.max(axis=1)
    y_hat = np.where(p2 >= 0.5, 2, 1)

    # --- Trade-off Acc vs Rejection
    thrs = np.linspace(thr_min, thr_max, n_thr)
    rejs, accs = [], []
    for t in thrs:
        acc_mask = conf >= t
        rej = 1.0 - acc_mask.mean()
        acc = np.mean(y_hat[acc_mask] == y_true[acc_mask]) if acc_mask.any() else np.nan
        rejs.append(rej)
        accs.append(acc)

    # Punto operativo
    op_mask = conf >= thr_used
    op_rej = float(np.mean(conf < thr_used))
    op_acc = float(np.mean(y_hat[op_mask] == y_true[op_mask])) if op_mask.any() else np.nan

    # --- Calibration
    y_bin = (y_true == 2).astype(int)
    frac_pos, mean_pred = calibration_curve(y_bin, p2, n_bins=n_bins, strategy="uniform")
    ece = expected_calibration_error(y_bin, p2, n_bins=n_bins)

    # --- Plot (2 subplots)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=160)

    # Left: tradeoff
    ax = axes[0]
    ax.plot(rejs, accs, marker="o", markersize=3)
    ax.scatter([op_rej], [op_acc], s=70)
    ax.text(op_rej, op_acc, f"  thr={thr_used:.3f}", va="center")
    ax.set_xlabel("Rejection rate")
    ax.set_ylabel("Accuracy en aceptados")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Trade-off Accuracy vs Rejection")

    # Right: calibration
    ax = axes[1]
    ax.plot([0, 1], [0, 1])
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.set_xlabel("Probabilidad predicha (clase 2)")
    ax.set_ylabel("Frecuencia real (clase 2)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Calibration (ECE={ece:.4f})")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    return ece


def plot_confidence_hist(conf, y_true, y_pred_rej, y_pred_forced, out_png, title):
    """
    Histograma de confianza:
    - aceptados correctos
    - aceptados incorrectos
    - rechazados
    """
    conf = np.asarray(conf)
    y_true = np.asarray(y_true).astype(int)
    y_pred_rej = np.asarray(y_pred_rej).astype(int)
    y_pred_forced = np.asarray(y_pred_forced).astype(int)

    rej = (y_pred_rej == 0)
    acc = ~rej
    correct = acc & (y_pred_forced == y_true)
    wrong = acc & (y_pred_forced != y_true)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    bins = 20
    ax.hist(conf[correct], bins=bins, alpha=0.7, label="Aceptado & Correcto")
    ax.hist(conf[wrong],   bins=bins, alpha=0.7, label="Aceptado & Incorrecto")
    ax.hist(conf[rej],     bins=bins, alpha=0.7, label="Rechazado")
    ax.set_xlabel("Confianza (max proba)")
    ax.set_ylabel("Conteo")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(y_true, proba_12, out_png, title, n_bins=10):
    """
    Reliability diagram (binario, clase positiva = 2) + ECE.
    """
    y_true = np.asarray(y_true).astype(int)
    y_bin = (y_true == 2).astype(int)
    p2 = np.asarray(proba_12[:, 1]).astype(float)

    frac_pos, mean_pred = calibration_curve(y_bin, p2, n_bins=n_bins, strategy="uniform")
    ece = expected_calibration_error(y_bin, p2, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=160)
    ax.plot([0, 1], [0, 1])
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.set_xlabel("Probabilidad predicha (clase 2)")
    ax.set_ylabel("Frecuencia real (clase 2)")
    ax.set_title(f"{title}\nECE={ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return ece


def plot_expert_contrib(ens, X, y_true, out_png, title):
    """
    Gráfico de aporte:
    - meta_weights (del pkl)  -> eje izquierdo
    - BA por experto (holdout)-> eje derecho
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import OrderedDict

    y_true = np.asarray(y_true).astype(int)
    experts = OrderedDict((name, pipe) for name, pipe in ens["models"])
    expert_names = ens.get("expert_names", list(experts.keys()))
    meta_w = ens.get("meta_weights", {})

    names, weights, bas = [], [], []

    # Calcula BA por experto y toma peso del meta
    for nm in expert_names:
        if nm not in experts:
            continue
        mdl = experts[nm]
        try:
            y_hat = mdl.predict(X).astype(int)
            ba = balanced_accuracy_safe(y_true, y_hat, labels=(1, 2))
        except Exception:
            ba = np.nan

        names.append(nm)
        weights.append(float(meta_w.get(nm, 0.0)))
        bas.append(float(ba))

    if len(names) == 0:
        return

    x = np.arange(len(names))

    fig, ax1 = plt.subplots(figsize=(9.5, 4.8), dpi=160)

    # --- Eje 1: pesos meta
    bars1 = ax1.bar(
        x - 0.20, weights, width=0.40,
        color="tab:blue", alpha=0.85,
        label="meta_weights"
    )
    ax1.set_ylabel("Peso del meta")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)

    # --- Eje 2: BA experto
    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + 0.20, bas, width=0.40,
        color="tab:orange", alpha=0.85,
        label="BA experto (holdout)"
    )
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_ylim(0, 1)

    ax1.set_title(title)

    # --- Leyenda combinada (una sola)
    handles = [bars1, bars2]
    labels = ["meta_weights", "BA experto (holdout)"]
    ax1.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)



def plot_channel_importance(ens, out_png, title):
    """
    Si existe ens['feature_importance'] (del RF), graficar importancia por canal.
    """
    fi = ens.get("feature_importance", None)
    if fi is None:
        return
    chans = np.asarray(fi.get("channels", []))
    imps = np.asarray(fi.get("importance", []), dtype=float)
    if chans.size == 0 or imps.size == 0 or chans.size != imps.size:
        return

    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=160)
    ax.bar(np.arange(len(chans)), imps)
    ax.set_xticks(np.arange(len(chans)))
    ax.set_xticklabels([str(int(c)) for c in chans], rotation=0)
    ax.set_xlabel("Canal (0-based)")
    ax.set_ylabel("Importancia media (RF)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 5) Procesamiento por sujeto + global
# =========================================================

def normalize_channels_zero_based(channels, n_ch=24):
    """
    En tu entrenamiento, channels son 0..23 (0-based).
    Si por alguna razón vienen 1..24, se corrige.
    """
    ch = np.asarray(channels).astype(int).ravel()
    if ch.size == 0:
        return np.arange(n_ch, dtype=int)

    # caso 1: 0..n-1
    if ch.min() >= 0 and ch.max() <= n_ch - 1:
        return ch

    # caso 2: 1..n
    if ch.min() >= 1 and ch.max() <= n_ch:
        return ch - 1

    # fallback: clamp a rango
    ch = np.clip(ch, 0, n_ch - 1)
    return ch


def run_subject(subj: str):
    subj_dir = BASE / subj
    if not subj_dir.exists():
        print(f"[{subj}] carpeta no encontrada.")
        return None

    pkl_path = subj_dir / f"{subj}_model_ensemble.pkl"
    mat_path = subj_dir / f"{subj}_holdout25.mat"
    if not pkl_path.exists() or not mat_path.exists():
        print(f"[{subj}] faltan archivos .pkl o .mat.")
        return None

    out_dir = OUT / subj
    out_dir.mkdir(parents=True, exist_ok=True)

    ens = load_ensemble(pkl_path)
    m = sio.loadmat(mat_path)
    O = m["O"]
    R = m["R"]
    y_true = m["labels"].squeeze().astype(int)

    # t_mask y channels 0-based (como tu entrenamiento con np.arange(24))
    t_mask = ens.get("t_mask", np.ones(O.shape[1], dtype=bool))
    channels = ens.get("channels", None)
    if channels is None:
        # fallback si no existe
        channels = ens.get("feature_importance", {}).get("channels", np.arange(24))
    channels = normalize_channels_zero_based(channels, n_ch=O.shape[2])

    # features iguales al entrenamiento: N x (len(channels)*7)
    X = features(O, R, t_mask, channels)

    # predicción meta + rechazo
    pred = predict_meta_with_rejection(ens, X, y_true)
    proba_12 = pred["proba_12"]
    conf = pred["conf"]
    thr_used = pred["thr_used"]
    y_forced = pred["y_pred_forced"]
    y_rej = pred["y_pred_rej"]

    # --- métricas
    rej_mask = (y_rej == 0)
    acc_mask = ~rej_mask
    rej_rate = float(rej_mask.mean())
    cov = float(acc_mask.mean())

    acc_accept = float(np.mean(y_rej[acc_mask] == y_true[acc_mask])) if acc_mask.any() else np.nan
    ba_accept  = balanced_accuracy_safe(y_true[acc_mask], y_rej[acc_mask]) if acc_mask.any() else np.nan

    acc_forced = float(np.mean(y_forced == y_true))
    ba_forced  = balanced_accuracy_safe(y_true, y_forced)

    # --- matrices de confusión
    cm_with = confusion_matrix(y_true, y_rej, labels=LABELS_3)
    cm_forced = confusion_matrix(y_true, y_forced, labels=LABELS_2)
    cm_accept = confusion_matrix(y_true[acc_mask], y_rej[acc_mask], labels=LABELS_2) if acc_mask.any() else np.zeros((2,2), dtype=int)

    plot_confusion(cm_with, NAMES_3, f"{subj} · Matriz de confusión (con rechazo)", out_dir / f"{subj}_confusion_WITH_rejection.png")
    plot_confusion(cm_forced, NAMES_2, f"{subj} · Matriz (sin rechazo, FORCED: todas las pruebas)", out_dir / f"{subj}_confusion_NO_rejection_FORCED.png")
    plot_confusion(cm_accept, NAMES_2, f"{subj} · Matriz (solo aceptados: suma={acc_mask.sum()})", out_dir / f"{subj}_confusion_NO_rejection_ACCEPTED_ONLY.png")

    # trade-off
    ece = plot_tradeoff_acc_and_calibration(
    y_true=y_true,
    proba_12=proba_12,
    thr_used=thr_used,
    out_png=out_dir / f"{subj}_tradeoff_plus_calibration.png",
    title=f"{subj} · Trade-off + Calibration"
)

    # hist confianza
    plot_confidence_hist(conf, y_true, y_rej, y_forced, out_dir / f"{subj}_confidence_hist.png",
                         title=f"{subj} · Confianza (correcto/incorrecto/rechazado)")

    # aporte de expertos
    plot_expert_contrib(ens, X, y_true, out_dir / f"{subj}_experts_contribution.png",
                        title=f"{subj} · Aporte expertos (peso meta vs BA)")

    # importancia canal (si existe)
    plot_channel_importance(ens, out_dir / f"{subj}_channel_importance.png",
                            title=f"{subj} · Importancia por canal (RF)")

    # guardar CSV por sujeto
    row = dict(
        subject=subj,
        n_test=int(len(y_true)),
        thr_used=float(thr_used),
        rejection_rate=rej_rate,
        coverage=cov,
        acc_accept=acc_accept,
        ba_accept=ba_accept,
        acc_forced=acc_forced,
        ba_forced=ba_forced,
        ece=ece,
        experts_kept="|".join(ens.get("experts_kept", ens.get("expert_names", [])))
    )
    pd.DataFrame([row]).to_csv(out_dir / f"{subj}_metrics.csv", index=False)

    # devolver lo necesario para global
    return dict(
        subj=subj,
        y_true=y_true,
        y_forced=y_forced,
        y_rej=y_rej,
        proba_12=proba_12,
        conf=conf,
        thr_used=thr_used,
        metrics=row,
        meta_weights=ens.get("meta_weights", {}),
        experts_kept=ens.get("experts_kept", ens.get("expert_names", []))
    )


def plot_global_performance_panel(df, out_png):
    """
    Panel global por sujeto:
    - barra: BA aceptados
    - línea: rechazo
    """
    d = df.copy().sort_values("ba_accept", ascending=False)
    x = np.arange(len(d))

    fig, ax1 = plt.subplots(figsize=(12.5, 5.0), dpi=160)
    ax1.bar(x, d["ba_accept"].values)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("BA (aceptados)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(d["subject"].values, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, d["rejection_rate"].values, marker="o")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Rejection rate")

    ax1.set_title("Global · Desempeño por sujeto (BA aceptados + rechazo)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def run_all():
    OUT.mkdir(parents=True, exist_ok=True)
    global_dir = OUT / "GLOBAL"
    global_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for subj in SUBJECTS:
        print(f"\n=== {subj} ===")
        try:
            r = run_subject(subj)
            if r is not None:
                results.append(r)
                print(f"[OK] {subj}")
        except Exception as e:
            print(f"[FAIL] {subj}: {e}")

    if len(results) == 0:
        print("No hubo sujetos procesados.")
        return

    # --- Global confusions (conteos)
    y_true_all = np.concatenate([r["y_true"] for r in results])
    y_forced_all = np.concatenate([r["y_forced"] for r in results])
    y_rej_all = np.concatenate([r["y_rej"] for r in results])

    cm_global_with = confusion_matrix(y_true_all, y_rej_all, labels=LABELS_3)
    cm_global_forced = confusion_matrix(y_true_all, y_forced_all, labels=LABELS_2)

    acc_mask_all = (y_rej_all != 0)
    cm_global_accept = confusion_matrix(y_true_all[acc_mask_all], y_rej_all[acc_mask_all], labels=LABELS_2) if acc_mask_all.any() else np.zeros((2,2), dtype=int)

    plot_confusion(cm_global_with, NAMES_3, "GLOBAL · Matriz (con rechazo)", global_dir / "GLOBAL_confusion_WITH_rejection.png")
    plot_confusion(cm_global_forced, NAMES_2, "GLOBAL · Matriz (sin rechazo, FORCED)", global_dir / "GLOBAL_confusion_NO_rejection_FORCED.png")
    plot_confusion(cm_global_accept, NAMES_2, f"GLOBAL · Matriz (solo aceptados: suma={acc_mask_all.sum()})", global_dir / "GLOBAL_confusion_NO_rejection_ACCEPTED_ONLY.png")

    # --- Global trade-off (pooling)
    conf_all = np.concatenate([r["conf"] for r in results])
    proba_all = np.vstack([r["proba_12"] for r in results])
    # usar promedio de thr_used sólo como marcador (opcional)
    thr_used_mean = float(np.mean([r["thr_used"] for r in results]))

    # --- Global confidence histogram
    plot_confidence_hist(conf_all, y_true_all, y_rej_all, y_forced_all,
                         global_dir / "GLOBAL_confidence_hist.png",
                         title="GLOBAL · Confianza (correcto/incorrecto/rechazado)")

    # --- Global reliability + ECE
    ece_global = plot_tradeoff_acc_and_calibration(
    y_true=y_true_all,
    proba_12=proba_all,
    thr_used=thr_used_mean,
    out_png=global_dir / "GLOBAL_tradeoff_plus_calibration.png",
    title="GLOBAL · Trade-off + Calibration"
)

    # --- CSV resumen
    df = pd.DataFrame([r["metrics"] for r in results])
    df.to_csv(global_dir / "GLOBAL_summary_metrics.csv", index=False)

    # --- Panel desempeño por sujeto
    plot_global_performance_panel(df, global_dir / "GLOBAL_performance_panel.png")

    # --- Expert stats global: frecuencia experts_kept + meta_weights promedio
    kept_counter = Counter()
    weights_rows = []
    for r in results:
        kept_counter.update(r["experts_kept"])
        weights_rows.append(r["meta_weights"])

    # frecuencia kept
    kept_names = sorted(kept_counter.keys())
    kept_vals = np.array([kept_counter[k] for k in kept_names], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=160)
    x_idx = np.arange(len(kept_names))
    # palette: cycle through a tab10 colormap so bars have distinct colors
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(kept_names))]
    bars = ax.bar(x_idx, kept_vals, color=colors, alpha=0.9)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(kept_names)
    ax.set_ylabel("# sujetos donde quedó")
    ax.set_title("GLOBAL · Frecuencia de experts_kept")
    # add exact value labels above each bar
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h, f"{int(h)}",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(global_dir / "GLOBAL_experts_kept_frequency.png", bbox_inches="tight")
    plt.close(fig)

    # meta_weights promedio
    # normalizamos a columnas comunes
    all_experts = sorted({k for w in weights_rows for k in w.keys()})
    W = np.zeros((len(weights_rows), len(all_experts)), dtype=float)
    for i, w in enumerate(weights_rows):
        for j, nm in enumerate(all_experts):
            W[i, j] = float(w.get(nm, 0.0))

    w_mean = W.mean(0)
    w_std = W.std(0)

    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=160)
    x_idx = np.arange(len(all_experts))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(all_experts))]
    bars = ax.bar(x_idx, w_mean, color=colors, alpha=0.9)
    ax.errorbar(x_idx, w_mean, yerr=w_std, fmt="none", capsize=4, ecolor="black")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(all_experts)
    ax.set_ylabel("Peso promedio")
    ax.set_title("GLOBAL · meta_weights (promedio ± std)")
    # add numeric labels above bars with 2 decimals
    for rect, m in zip(bars, w_mean):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h, f"{m:.2f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(global_dir / "GLOBAL_meta_weights_mean_std.png", bbox_inches="tight")
    plt.close(fig)

    print("\n[DONE] Reporte global en:", str(global_dir))
    print("ECE global:", ece_global)


if __name__ == "__main__":
    run_all()
