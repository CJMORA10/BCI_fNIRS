# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#  BCI-fNIRS · Comité + Stacking (V4 adaptativa + rechazo + gráficas)
#  - Umbral de rechazo ADAPTATIVO por sujeto (optimiza BA en aceptados)
#  - Poda de expertos débiles basada en OOF (meta-weights y BA por experto)
#  - Opción de usar LOGIT-PAIRS como features del meta (mejora separabilidad)
#  - RFECV y selección con 'balanced_accuracy'
#  - Registra CSV por sujeto + matrices de confusión con rechazo (PNG/CSV)
#  - Violin plots de 7 features por sujeto y gráficos globales
# ──────────────────────────────────────────────────────────────────────────────
import os
from joblib.externals.loky import set_loky_pickler

# ---------- CONTROL DE HILOS GLOBALES ----------
os.environ["OMP_NUM_THREADS"] = "1"       # OpenMP / BLAS
os.environ["MKL_NUM_THREADS"] = "1"       # MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
set_loky_pickler("pickle")
N_JOBS = 4

import numpy as np, scipy.io as sio, pandas as pd, warnings, pickle, statistics, random
from pathlib import Path
from collections import defaultdict
from joblib import parallel_backend

from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit, cross_val_predict, cross_val_score)
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import (RFECV, RFE, SelectKBest, f_classif)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from boruta import BorutaPy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel

# --- Plotting (backend no interactivo) --- 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════════════════
SEED = 0
np.random.seed(SEED); random.seed(SEED)

DATA_DIR  = Path("SEGMENTATION_TASK_MI")
OUTPUT    = Path("TRAINED_MODEL_IMPROVED_V4_FINAL"); OUTPUT.mkdir(exist_ok=True, parents=True)
SUBJECTS  = [f"subject {i:02}" for i in range(1, 29)]

N_FOLDS, N_SPLITS, TEST_SIZE = 5, 15, 0.25
K_MIN, MIN_HEMI              = 6, 3
R2_CANDS                     = [0.05, 0.04, 0.03]

MIN_FEAT, MAX_FEAT = 6, 10

# Adaptación de rechazo y poda
TARGET_REJ_BASE = 0.10
REJ_RANGE = (0.05, 0.18)                       # rango permitido para optimización (OOF)
TARGET_GRID = [0.08, 0.10, 0.12, 0.15]         # objetivos candidatos
PRUNE_WEAK_EXPERTS = True
WEAK_WEIGHT_THR = 0.08                         # peso meta mínimo (OOF)
WEAK_BA_THR = 0.55                              # BA OOF mínimo por experto
USE_LOGIT_META = True                           # usar log-odds por experto en el meta

CATALOG = {"A": (-5, 0),  "B": (0, 10), "C": (5, 8),   "D": (10, 13),
           "E": (15, 18), "F": (16, 19), "G": (-5, -2), "H": (17, 20),
           "I": (2,12), "J": (3,13), "K": (4,14), "L": (5,15)}
HEMISPHERE = {'L': np.arange(0, 12), 'R': np.arange(12, 24)}

LABEL_NAMES = {1: "Mano derecha", 2: "Mano izquierda", 0: "Rechazo"}

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def r2_signed(epo, y):
    cls = [epo[y == k] for k in (1, 2)]
    mu  = [c.mean(0) for c in cls]
    var = [c.var(0, ddof=1) for c in cls]
    r2  = (mu[0] - mu[1])**2 / (var[0] + var[1] + 1e-12)
    return r2 * np.where(mu[0] > mu[1], 1, -1)

def slope(x): return np.diff(x, axis=1).mean(1)

def features(O, R, t_mask, ch):
    O_ = O[:, t_mask][:, :, ch]
    R_ = R[:, t_mask][:, :, ch]
    f = [O_.mean(1), slope(O_), R_.mean(1), slope(R_), O_.max(1), O_.var(1), (O_ - R_).mean(1)]
    F = np.stack(f, axis=2)
    return F.reshape(O_.shape[0], -1)

FEAT_PER_CH = 7

def cols_from_channels(ch_sel):
    return np.concatenate([np.arange(c*FEAT_PER_CH,(c+1)*FEAT_PER_CH) for c in ch_sel])

FEATURE_LABELS = ["O_mean","O_slope","R_mean","R_slope","O_peak","O_var","O_minus_R_mean"]

def make_feature_names_for_cols(ch_sel):
    return [f"ch{int(c)}:{lab}" for c in ch_sel for lab in FEATURE_LABELS]

# Selectors / pipeline utilities
def get_selector_from_pipeline(pipe):
    if hasattr(pipe, "named_steps"):
        for k in ("fs", "boruta"):
            if k in pipe.named_steps:
                return pipe.named_steps[k], k
    return None, None

def get_selected_indices_from_selector(selector, n_features):
    if selector is None:
        return np.arange(n_features)
    if hasattr(selector, "get_support"):
        return selector.get_support(indices=True)
    return np.arange(n_features)

def get_selector_details(selector, n_features):
    sel_idx = get_selected_indices_from_selector(selector, n_features)
    details = {"selected_idx": sel_idx}
    if hasattr(selector, "ranking_"):
        details["ranking"] = selector.ranking_.tolist()
    return details

# Wrappers
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
    def __init__(self, estimator, n_estimators='auto', perc=100, alpha=0.05, max_iter=100, random_state=None, verbose=0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.boruta = None
    def fit(self, X, y):
        self.boruta = BorutaPy(self.estimator, n_estimators=self.n_estimators, perc=self.perc, alpha=self.alpha, max_iter=self.max_iter, random_state=self.random_state, verbose=self.verbose)
        self.boruta.fit(X, y)
        if not any(self.boruta.support_):
            n_features_total = X.shape[1]
            n_features = min(10, max(3, int(0.3 * n_features_total)))
            temp_rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=1, class_weight='balanced')
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
    def __init__(self, estimator, min_features=MIN_FEAT, threshold=None, prefit=False, norm_order=1, max_features=MAX_FEAT, importance_getter='auto'):
        self.estimator = estimator
        self.min_features = min_features
        self.threshold = threshold
        self.prefit = prefit
        self.norm_order = norm_order
        self.max_features = max_features
        self.importance_getter = importance_getter
        self.selector = None
    def fit(self, X, y):
        self.selector = SelectFromModel(self.estimator, threshold=self.threshold, prefit=self.prefit, norm_order=self.norm_order, max_features=self.max_features, importance_getter=self.importance_getter)
        self.selector.fit(X, y)
        support = self.selector.get_support(); n_selected = support.sum()
        if n_selected < self.min_features:
            if hasattr(self.selector.estimator_, 'coef_'):
                importances = np.abs(self.selector.estimator_.coef_).ravel()
            elif hasattr(self.selector.estimator_, 'feature_importances_'):
                importances = self.selector.estimator_.feature_importances_
            else:
                importances = np.arange(X.shape[1])
            n_to_select = min(self.min_features, X.shape[1])
            top_indices = np.argsort(importances)[-n_to_select:]
            self.support_ = np.zeros(X.shape[1], dtype=bool); self.support_[top_indices] = True
        else:
            self.support_ = support
        return self
    def transform(self, X):
        return X[:, self.support_]
    def get_support(self, indices=False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_

# Metrics
def balanced_accuracy_safe(y_true, y_pred, labels=(1,2)):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    recalls = []
    for c in labels:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = tp + fn
        recalls.append(0.5 if denom == 0 else tp/denom)
    return float(np.mean(recalls))

# Threshold search
def find_best_threshold(y_true, proba, thr_min=0.50, thr_max=0.95, n=31, target_rej=None, rej_bounds=None):
    conf = proba.max(1); y_hat = proba.argmax(1) + 1
    thrs = np.linspace(thr_min, thr_max, n)
    best = (thr_min, -np.inf, 0.0, 0.0)
    for t in thrs:
        mask = conf >= t
        if mask.sum() == 0:
            continue
        ba  = balanced_accuracy_safe(y_true[mask], y_hat[mask])
        rej = 1 - mask.mean()
        if rej_bounds is not None:
            lo, hi = rej_bounds
            if not (lo <= rej <= hi):
                continue
        util = ba if target_rej is None else (ba - abs(rej - target_rej))
        if util > best[1]:
            best = (float(t), float(util), float(rej), float(ba))
    return best  # (thr, util, rej_rate, ba_on_accepted)

# Meta utilities
def probs_to_logit_pairs(Z, eps=1e-6):
    Z = np.clip(Z, eps, 1-eps)
    exps = Z.shape[1]//2
    feats = []
    for i in range(exps):
        p1 = Z[:, 2*i]     # prob clase 1
        p2 = Z[:, 2*i+1]   # prob clase 2
        feats.append(np.log(p2/p1))  # log-odds
    return np.column_stack(feats)

def meta_expert_weights_from_OOF(Z, y, expert_names):
    n_exp = len(expert_names)
    lr = LogisticRegression(max_iter=5000, class_weight='balanced', solver='lbfgs')
    lr.fit(Z, y)
    coef = np.asarray(lr.coef_).ravel(); p = Z.shape[1]
    if p == 2*n_exp:
        w = [np.sum(np.abs(coef[2*i:2*i+2])) for i in range(n_exp)]
    else:
        block = max(1, p // n_exp); w = []
        for i in range(n_exp):
            start = i*block; end = min(start+block, p)
            w.append(float(np.sum(np.abs(coef[start:end]))))
    w = np.asarray(w, dtype=float)
    if not np.isfinite(w).any() or w.sum() <= 0:
        w = np.ones(n_exp, dtype=float) / n_exp
    else:
        w = w / w.sum()
    return {expert_names[i]: float(w[i]) for i in range(n_exp)}

# RF feature importance → canales
def get_rf_pipeline_from_models(models):
    for name, est in models:
        if name == "rf" and hasattr(est, "named_steps") and "rf" in est.named_steps:
            return est
    return None

def analyze_feature_importance_best(model_pkg, n_cols_split):
    channels = model_pkg['channels']
    rf_pipe = get_rf_pipeline_from_models(model_pkg['models'])
    if rf_pipe is None: return None
    rf = rf_pipe.named_steps['rf']
    importances = getattr(rf, "feature_importances_", None)
    if importances is None: return None
    selector, _ = get_selector_from_pipeline(rf_pipe)
    sel_idx = selector.get_support(indices=True) if (selector is not None and hasattr(selector, "get_support")) else np.arange(n_cols_split)
    importances = np.asarray(importances).ravel()
    if len(importances) != len(sel_idx):
        m = min(len(importances), len(sel_idx)); importances = importances[:m]; sel_idx = sel_idx[:m]
    full_imp = np.zeros(n_cols_split, dtype=float); full_imp[sel_idx] = importances
    n_ch = len(channels); ch_imp = np.zeros(n_ch, dtype=float)
    for i in range(n_ch):
        start = i*FEAT_PER_CH; end = start+FEAT_PER_CH
        ch_imp[i] = full_imp[start:end].mean()
    top_local_idx = np.argsort(-ch_imp)[:5]
    return dict(
        channels=channels.tolist(),
        importance=ch_imp.tolist(),
        top_channels=(channels[top_local_idx]).tolist(),
        top_importance=(ch_imp[top_local_idx]).tolist()
    )

# Resumen top features por experto
def summarize_best_features_by_model(trained_models, feat_names):
    out = {}; p = len(feat_names)
    for name, est in trained_models:
        pipe = est
        if not hasattr(pipe, "named_steps"): continue
        selector, _ = get_selector_from_pipeline(pipe)
        sel_idx = get_selected_indices_from_selector(selector, p)
        model_imp = None
        if "rf" in pipe.named_steps and hasattr(pipe.named_steps["rf"], "feature_importances_"):
            fi = pipe.named_steps["rf"].feature_importances_; model_imp = np.zeros(p); model_imp[sel_idx[:len(fi)]] = fi
        if "xgb" in pipe.named_steps and hasattr(pipe.named_steps["xgb"], "feature_importances_"):
            fi = pipe.named_steps["xgb"].feature_importances_; model_imp = np.zeros(p); model_imp[sel_idx[:len(fi)]] = fi
        scores = getattr(selector, "scores_", None); ranking = getattr(selector, "ranking_", None)
        if model_imp is not None and model_imp.sum() > 0:
            order = np.argsort(-model_imp); top_idx = [i for i in order if model_imp[i] > 0][:10]; crit = "importancia_modelo"
        elif scores is not None:
            order = np.argsort(-scores); top_idx = [i for i in order if i in sel_idx][:10]; crit = "scores_SelectKBest"
        elif ranking is not None:
            order = np.argsort(ranking); top_idx = [i for i in order if i in sel_idx][:10]; crit = "ranking_RFE"
        else:
            top_idx = sel_idx[:10]; crit = "seleccion_sin_orden"
        out[name] = dict(criterion=crit, top_features=[feat_names[i] for i in top_idx], selected_count=int(len(sel_idx)))
    return out

# Aux: columnas por experto en Z
def expert_col_indices(names, total_cols):
    # Asume 2 columnas por experto (p1,p2) en el stacking Z
    idx_map = {}
    for i, nm in enumerate(names):
        a, b = 2*i, 2*i+1
        if b < total_cols:
            idx_map[nm] = (a,b)
    return idx_map

# ── Visualización ─────────────────────────────────────────────────────────────
def plot_confusion_png(cm, title, out_png):
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    disp = ConfusionMatrixDisplay(cm, display_labels=[LABEL_NAMES[1], LABEL_NAMES[2], LABEL_NAMES[0]])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# Colapsa N x (ch * 7) → N x 7, promediando sobre canales seleccionados
def collapse_features_7(X_sel, n_channels, feat_per_ch=FEAT_PER_CH):
    Xr = X_sel.reshape(X_sel.shape[0], n_channels, feat_per_ch)  # [N, C, 7]
    return Xr.mean(axis=1)  # [N, 7]

def violin_features_png(X7, y, feature_labels, title, out_png):
    data1 = [X7[y == 1, i] for i in range(X7.shape[1])]
    data2 = [X7[y == 2, i] for i in range(X7.shape[1])]
    pos1 = np.arange(1, X7.shape[1]+1) - 0.15
    pos2 = np.arange(1, X7.shape[1]+1) + 0.15

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    ax.violinplot(data1, positions=pos1, widths=0.25, showmeans=True, showextrema=False, showmedians=False)
    ax.violinplot(data2, positions=pos2, widths=0.25, showmeans=True, showextrema=False, showmedians=False)
    ax.set_xticks(np.arange(1, X7.shape[1]+1))
    ax.set_xticklabels(feature_labels, rotation=30, ha="right")
    ax.set_xlim(0.5, X7.shape[1]+0.5)
    ax.set_title(title)
    ax.set_ylabel("Valor de la característica")
    # pseudo-leyenda
    ax.plot([], [], label=LABEL_NAMES[1])
    ax.plot([], [], label=LABEL_NAMES[2])
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_global_performance(df, out_png):
    d = df.copy()
    d["subject"] = d["subject"].astype(str)
    d = d.sort_values("acc_holdout", ascending=False)
    x = np.arange(len(d))
    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=160)

    ax1.bar(x - 0.2, d["acc_holdout"].values, width=0.4, label="Accuracy Hold-out")
    ax1.bar(x + 0.2, d["accuracy_no_rejection"].values, width=0.4, label="Accuracy en aceptados")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(d["subject"].values, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, d["rej_rate"].values, marker="o", label="Tasa de rechazo")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Rechazo")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    ax1.set_title("Desempeño por sujeto (con rechazo)")
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# ACUMULADORES GLOBALES
# ═════════════════════════════════════════════════════════════════════════════
global_cm = np.zeros((3, 3), dtype=int)  # matriz (1,2,0)x(1,2,0)
global_X7_list = []
global_y_list = []

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
summary, best_cfg = [], {}

for subj in SUBJECTS:
    print(f"\n{'='*60}\nProcesando {subj}\n{'='*60}")
    mat = DATA_DIR / subj / "Epochs_MI_classification.mat"
    if not mat.exists():
        warnings.warn(f"{subj}: .mat no encontrado - omitido"); continue

    m = sio.loadmat(mat, squeeze_me=True)
    O, R = m['epochs_O'], m['epochs_R']; y = m['labels'].astype(int).ravel()
    N, T, _ = O.shape
    print(f"Sólo {N} muestras totales")

    t_sec = np.linspace(-10, 25, T)
    VALID_KEYS = {'B','C','D','E','F','I','J','K','L'}
    masks = {k:(t_sec>=a)&(t_sec<=b) for k,(a,b) in CATALOG.items() if k in VALID_KEYS}

    best_model=None; best_thr_fin=None
    sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)
    hold_scores=[]; best_acc_export=-1; best_cv_mean=best_cv_std=0

    # placeholders para mejores artefactos del sujeto (se definen al encontrar best split)
    best_cm = None
    best_X7_all = None
    best_y_all = None
    best_out_dir = None

    with parallel_backend("loky", n_jobs=N_JOBS, inner_max_num_threads=1):
      for split_id,(idx_tr,idx_te) in enumerate(sss.split(np.zeros(N),y),1):
        O_tr,R_tr,y_tr = O[idx_tr],R[idx_tr],y[idx_tr]
        O_te,R_te,y_te = O[idx_te],R[idx_te],y[idx_te]
        print(f"\n  === Split {split_id}/{N_SPLITS} ===")
        print(f"    Datos: Train={len(y_tr)}, Test={len(y_te)}")

        X_win_tr = {k:features(O_tr,R_tr,m,np.arange(24)).astype(np.float32, copy=False) for k,m in masks.items()}
        X_win_te = {k:features(O_te,R_te,m,np.arange(24)).astype(np.float32, copy=False) for k,m in masks.items()}

        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        cfg_fold=defaultdict(list); acc_outer_split=[]

        # ── nested-CV ventana/canales ─────────────────────────────────
        for tr_idx,va_idx in cv.split(np.zeros(len(y_tr)),y_tr):
          best_acc=-1
          for k,t_mask in masks.items():
            r2_tr  = r2_signed(O_tr[tr_idx],y_tr[tr_idx])
            abs_r2 = np.abs(r2_tr[t_mask].mean(0))
            idx_sorted=np.argsort(-abs_r2)
            for thr in R2_CANDS:
              ch_sel=np.where(abs_r2>=thr)[0]
              if ch_sel.size<K_MIN:
                  extra=[i for i in idx_sorted if i not in ch_sel][:K_MIN-ch_sel.size]
                  ch_sel=np.hstack([ch_sel,extra])
              for side in ('L','R'):
                hemi=HEMISPHERE[side]; present=np.intersect1d(ch_sel,hemi)
                if present.size<MIN_HEMI:
                    extra=[i for i in idx_sorted if i in hemi and i not in ch_sel][:MIN_HEMI-present.size]
                    ch_sel=np.hstack([ch_sel,extra])
              cols=cols_from_channels(ch_sel)
              X_tr_=X_win_tr[k][tr_idx][:,cols]
              X_va_=X_win_tr[k][va_idx][:,cols]
              scaler=StandardScaler().fit(X_tr_)
              X_tr_s, X_va_s = scaler.transform(X_tr_), scaler.transform(X_va_)
              lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.3)
              sel=RFECV(lda,step=0.1,cv=3,n_jobs=1,min_features_to_select=MIN_FEAT,scoring='balanced_accuracy').fit(X_tr_s,y_tr[tr_idx])
              f_idx=sel.get_support(indices=True)
              try:
                bag = BalancedBaggingClassifier(estimator=lda, n_estimators=30, max_samples=0.6, max_features=0.8, bootstrap=True, bootstrap_features=True, n_jobs=1, sampling_strategy='auto', random_state=SEED)
                bag.fit(X_tr_s[:, f_idx], y_tr[tr_idx])
                y_va_hat = bag.predict(X_va_s[:, f_idx])
                acc = balanced_accuracy_safe(y_tr[va_idx], y_va_hat)
              except Exception:
                acc = 0.0
              if acc > best_acc:
                  best_acc=acc; best_key=k; best_thr=thr
                  best_ch,best_feat=ch_sel.copy(),f_idx.copy()
          cfg_fold['win_key'].append(best_key); cfg_fold['thr'].append(best_thr)
          cfg_fold['ch'].append(best_ch);       cfg_fold['feat'].append(best_feat)
          acc_outer_split.append(best_acc)

        cv_mean_split=np.mean(acc_outer_split); cv_std_split=np.std(acc_outer_split)

        # ── votación de canales ───────────────────────────────────────
        win_mode=statistics.mode(cfg_fold['win_key']); t_mask_win=masks[win_mode]
        thr_mode=statistics.mode(cfg_fold['thr'])
        ch_votes=np.concatenate(cfg_fold['ch'])
        uniq,cnt=np.unique(ch_votes,return_counts=True)
        ch_sel=uniq[cnt>=len(cfg_fold['ch'])/2]
        if ch_sel.size < K_MIN:
            extra=[u for u in uniq[np.argsort(-cnt)] if u not in ch_sel][:K_MIN-ch_sel.size]
            ch_sel=np.hstack([ch_sel,extra])
        cols_final=cols_from_channels(ch_sel)

        X_feat_tr = X_win_tr[win_mode][:,cols_final]
        X_feat_te = X_win_te[win_mode][:,cols_final]
        feat_names = make_feature_names_for_cols(ch_sel)
        print(f"    Ventana: {win_mode} {CATALOG[win_mode]}, Canales: {len(ch_sel)}")

        # ── Expertos (igual que V3, scoring BA) ───────────────────────
        lda_base=LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.3)
        pipe_bag = Pipeline([
            ('scale',StandardScaler()),
            ('fs',RFECV(lda_base,step=0.1,cv=5,n_jobs=1,min_features_to_select=MIN_FEAT,scoring='balanced_accuracy')),
            ('bag',BalancedBaggingClassifier(estimator=lda_base,n_estimators=30,max_samples=0.6,max_features=0.8,bootstrap=True,bootstrap_features=True, sampling_strategy='auto', n_jobs=N_JOBS, random_state=SEED))
        ])
        pipe_svm = Pipeline([
            ('scale',StandardScaler()),
            ('fs', RFE(LinearSVC(C=0.1, penalty='l2', dual=False), n_features_to_select=MAX_FEAT, step=0.2)),
            ('svc',SVC(kernel='rbf',C=1.0,gamma='scale',probability=True,class_weight='balanced'))
        ])
        pipe_xgb = Pipeline([
            ('fs', SafeSelectFromModel(XGBSafeBinary(n_estimators=60, max_depth=3, learning_rate=0.1,subsample=0.8, colsample_bytree=0.8, random_state=SEED), min_features=MIN_FEAT, threshold='median', importance_getter='feature_importances_')),
            ('xgb', XGBSafeBinary(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_alpha=0.1, reg_lambda=1, random_state=SEED))
        ])
        rf_base = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=6, min_samples_split=5, min_samples_leaf=3, max_features='sqrt', random_state=SEED, n_jobs=N_JOBS)
        pipe_rf = Pipeline([
            ('boruta', SafeBoruta(estimator=RandomForestClassifier(n_estimators=50, random_state=SEED), n_estimators='auto', verbose=0, random_state=SEED)),
            ('rf', rf_base)
        ])
        pipe_mlp = Pipeline([
            ('scale',StandardScaler()),
            ('fs', SelectKBest(f_classif, k=MAX_FEAT)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(20,), solver='adam', learning_rate_init=1e-3, alpha=0.001, early_stopping=True, validation_fraction=0.2, n_iter_no_change=15, max_iter=2000, random_state=SEED))
        ])

        experts_tab = {'bag': pipe_bag,'svm': pipe_svm,'xgb': pipe_xgb,'rf' : pipe_rf,'mlp': pipe_mlp}

        # ══════════════════════════════════════════════════════════════
        # OOF por experto (para poda) + stacking + umbral adaptativo
        # ══════════════════════════════════════════════════════════════
        X_tr_feat = X_feat_tr.astype(np.float32, copy=False)
        X_te_feat = X_feat_te.astype(np.float32, copy=False)

        cv_meta = StratifiedKFold(5, shuffle=True, random_state=SEED)
        Z_tr_list, names, trained_models = [], [], []
        expert_oof = {}  # proba OOF por experto
        expert_ba = {}
        for name, pipe in experts_tab.items():
            try:
                oof_proba = cross_val_predict(pipe, X_tr_feat, y_tr, cv=cv_meta, method='predict_proba', n_jobs=1)
                Z_tr_list.append(oof_proba); names.append(name)
                expert_oof[name] = oof_proba
                y_hat_i = oof_proba.argmax(1) + 1
                expert_ba[name] = balanced_accuracy_safe(y_tr, y_hat_i)
                pipe.fit(X_tr_feat, y_tr); trained_models.append((name, pipe))
            except Exception as e:
                print(f"    {name} fuera del stacking por error: {e}")
        if len(Z_tr_list) == 0: raise RuntimeError("Ningún experto disponible para stacking.")
        Z_tr = np.concatenate(Z_tr_list, axis=1)

        # Opcional: features del meta como logit-pairs
        Z_tr_meta = probs_to_logit_pairs(Z_tr) if USE_LOGIT_META else Z_tr

        # Pesos iniciales del meta (interpretables)
        meta_weights = meta_expert_weights_from_OOF(Z_tr_meta, y_tr, names)

        # ── PODA de expertos débiles (opcional) ───────────────────────
        keep = []
        for nm in names:
            w = meta_weights.get(nm, 0.0); ba_i = expert_ba.get(nm, 0.5)
            if (w >= WEAK_WEIGHT_THR) or (ba_i >= WEAK_BA_THR):
                keep.append(nm)
        if PRUNE_WEAK_EXPERTS and len(keep) >= 2 and len(keep) < len(names):
            print(f"    Poda de expertos → manteniendo: {keep}")
            # Subconjunto de columnas de Z_tr según keep
            idx_map = expert_col_indices(names, Z_tr.shape[1])
            cols_keep = []
            for nm in keep:
                a,b = idx_map[nm]; cols_keep.extend([a,b])
            Z_tr = Z_tr[:, cols_keep]
            Z_tr_meta = probs_to_logit_pairs(Z_tr) if USE_LOGIT_META else Z_tr
            # Recalcular pesos con subset
            meta_weights = meta_expert_weights_from_OOF(Z_tr_meta, y_tr, keep)
            names = keep
            # Filtrar trained_models y expert_oof
            trained_models = [(nm, est) for (nm, est) in trained_models if nm in keep]
        else:
            keep = names[:]

        # Meta calibrado (igual que antes, pero sobre Z_tr_meta actual)
        meta_oof = CalibratedClassifierCV(estimator=LogisticRegression(max_iter=5000, class_weight='balanced', solver='lbfgs'), method='sigmoid', cv=5)
        oof_meta_proba = cross_val_predict(meta_oof, Z_tr_meta, y_tr, method='predict_proba', cv=5, n_jobs=1)

        # Umbral adaptativo: buscar mejor BA en aceptados con restricción de rechazo
        conf_oof = oof_meta_proba.max(1)
        qthr = float(np.quantile(conf_oof, TARGET_REJ_BASE))
        thr_lo = max(0.50, qthr - 0.07); thr_hi = min(0.95, qthr + 0.07)
        best_thr, best_util, best_rej, best_ba = None, -np.inf, None, None
        for tgt in TARGET_GRID:
            thr, util, rej, ba = find_best_threshold(y_tr, oof_meta_proba, thr_min=thr_lo, thr_max=thr_hi, n=41, target_rej=tgt, rej_bounds=REJ_RANGE)
            if util > best_util:
                best_thr, best_util, best_rej, best_ba = thr, util, rej, ba
        if best_thr is None:  # fallback seguro
            best_thr, _, best_rej, best_ba = find_best_threshold(y_tr, oof_meta_proba, thr_min=0.50, thr_max=0.95, n=91, target_rej=TARGET_REJ_BASE, rej_bounds=REJ_RANGE)
        THR_REJECT = float(best_thr)
        print(f"    Umbral adaptativo: {THR_REJECT:.3f} | OOF: rechazo≈{best_rej:.2%}, BA_aceptados={best_ba:.3f}")

        # Entrenar meta final con subset y representación elegida
        meta = CalibratedClassifierCV(estimator=LogisticRegression(max_iter=5000, class_weight='balanced', solver='lbfgs'), method='sigmoid', cv=5)
        meta.fit(Z_tr_meta, y_tr)

        # TEST: ensamblar Z_te sólo con expertos mantendidos
        Z_te_list = []
        for nm, model in trained_models:
            Z_te_list.append(model.predict_proba(X_te_feat))
        Z_te_full = np.concatenate(Z_te_list, axis=1)
        Z_te_meta = probs_to_logit_pairs(Z_te_full) if USE_LOGIT_META else Z_te_full

        proba_final = meta.predict_proba(Z_te_meta)
        y_pred = proba_final.argmax(axis=1) + 1
        acc_hold = accuracy_score(y_te, y_pred)
        hold_scores.append(acc_hold)

        conf = proba_final.max(1)
        thr_used = THR_REJECT
        if abs(float(np.mean(conf < thr_used)) - best_rej) > 0.08:  # recentrado suave
            thr_used = float(np.quantile(conf, np.clip(best_rej, REJ_RANGE[0], REJ_RANGE[1])))

        y_pred_rej = proba_final.argmax(1) + 1
        y_pred_rej[conf < thr_used] = 0

        rej_rate = float(np.mean(y_pred_rej == 0))
        acc_no_rej = accuracy_score(y_te[y_pred_rej != 0], y_pred_rej[y_pred_rej != 0]) if np.any(y_pred_rej != 0) else np.nan

        # Por-clase
        mask1 = y_te == 1; mask2 = y_te == 2
        acc_c1 = accuracy_score(y_te[mask1], y_pred[mask1]) if mask1.any() else np.nan
        acc_c2 = accuracy_score(y_te[mask2], y_pred[mask2]) if mask2.any() else np.nan

        if acc_hold > best_acc_export:
            best_acc_export   = acc_hold
            best_cv_mean      = cv_mean_split
            best_cv_std       = cv_std_split
            best_accuracy_no_rejection = acc_no_rej
            best_rej_rate     = rej_rate
            best_acc_c1       = acc_c1
            best_acc_c2       = acc_c2

            per_model_feats = {}
            for nm, pipe in trained_models:
                selector, sel_name = get_selector_from_pipeline(pipe)
                details = get_selector_details(selector, n_features=X_feat_tr.shape[1])
                sel_idx = details["selected_idx"]
                sel_names = [feat_names[i] for i in sel_idx]
                per_model_feats[nm] = {"n_features": int(len(sel_idx)),"indices_local": sel_idx.tolist(),"feature_names": sel_names}
                if "ranking" in details: per_model_feats[nm]["ranking"] = details["ranking"]

            top_feats_by_model = summarize_best_features_by_model(trained_models, feat_names)
            importance_analysis = analyze_feature_importance_best(dict(models=trained_models, channels=ch_sel.copy()), X_feat_tr.shape[1])

            model_pkg = dict(
                models=trained_models,
                stacking_meta=meta,
                expert_names=names,
                t_mask = t_mask_win,
                n_samples_win = int(t_mask_win.sum()),
                channels=ch_sel.copy(),
                win_key=win_mode,
                win_sec=CATALOG[win_mode],
                stacking=True,
                accuracy=acc_hold,
                accuracy_no_rejection=acc_no_rej,
                feature_usage=per_model_feats,
                total_features=X_feat_tr.shape[1],
                thr_reject=float(THR_REJECT),
                target_rej=float(best_rej),
                meta_weights=meta_weights,
                top_features_by_model=top_feats_by_model,
                experts_kept=names
            )
            if importance_analysis: model_pkg['feature_importance'] = importance_analysis

            best_model, best_thr_fin = model_pkg, thr_mode
            out_dir = OUTPUT/subj; out_dir.mkdir(exist_ok=True,parents=True)
            with open(out_dir/f"{subj}_model_ensemble.pkl","wb") as f:
                pickle.dump(model_pkg,f)
            sio.savemat(out_dir/f"{subj}_holdout25.mat", {"O":O_te,"R":R_te,"labels":y_te})

            # ── NUEVO: guardar artefactos del mejor split para plots por sujeto
            cm_curr = confusion_matrix(y_te, y_pred_rej, labels=[1, 2, 0])
            best_cm = cm_curr.copy()
            # Violin: usar TODAS las épocas del sujeto en la ventana ganadora y canales elegidos
            X_all_win = features(O, R, masks[win_mode], np.arange(24)).astype(np.float32, copy=False)
            X_sel_all = X_all_win[:, cols_final]
            X7_all    = collapse_features_7(X_sel_all, n_channels=len(ch_sel))
            best_X7_all = X7_all.copy()
            best_y_all  = y.copy()
            best_out_dir = out_dir

    # Limpieza
    import gc
    if 'X_feat_tr' in locals(): del X_feat_tr
    if 'X_feat_te' in locals(): del X_feat_te
    gc.collect()

    if best_model is None:
        warnings.warn(f"{subj}: no se pudo fijar un mejor split — se omite."); continue

    print(f"\n{subj} RESUMEN:")
    print(f"  CV: {best_cv_mean:.3f}±{best_cv_std:.3f}")
    print(f"  Mejor Hold-out: {best_acc_export:.3f}")
    print(f"  Rendimiento sin rechazos: {best_accuracy_no_rejection:.3f}")
    print(f"  Tasa de rechazo: {best_rej_rate:.2%}")
    print(f"  Acc Clase 1: {best_acc_c1:.3f}")
    print(f"  Acc Clase 2: {best_acc_c2:.3f}")

    bm=best_model
    row=dict(
        win_key=bm['win_key'],
        win_sec_start=bm['win_sec'][0],
        win_sec_end=bm['win_sec'][1],
        r2_thr=best_thr_fin,
        n_channels=len(bm['channels']),
        channels="|".join(map(str, bm['channels'].tolist())),
        CV_mean=best_cv_mean, CV_std=best_cv_std,
        acc_holdout=best_acc_export,
        acc_holdout_mean=np.mean(hold_scores) if len(hold_scores) else np.nan,
        acc_holdout_std=np.std(hold_scores) if len(hold_scores) else np.nan,
        accuracy_no_rejection = best_accuracy_no_rejection,
        rej_rate = best_rej_rate,
        acc_class1 = best_acc_c1,
        acc_class2 = best_acc_c2,
        thr_reject = bm.get("thr_reject", np.nan),
        target_rej = bm.get("target_rej", np.nan),
        experts_kept = "|".join(bm.get("experts_kept", []))
    )

    # Importancias por canal
    fi = bm.get("feature_importance")
    if fi is not None:
        row["top_channels_ids"] = "|".join(map(str, fi["top_channels"]))
        row["top_channels_imp"] = "|".join(f"{x:.6f}" for x in fi["top_importance"])
        row["channels_importance"] = "|".join(f"{x:.6f}" for x in fi["importance"])

    # Pesos del meta
    mw = bm.get("meta_weights", {})
    for k, v in mw.items(): row[f"meta_w_{k}"] = float(v)

    # Top features por experto
    tf = bm.get("top_features_by_model", {})
    for m, info in tf.items():
        row[f"{m}_top10_criterion"] = info.get("criterion", "")
        row[f"{m}_top10"] = "|".join(info.get("top_features", []))
        row[f"{m}_selected_count"] = info.get("selected_count", np.nan)

    best_cfg[subj]=row

    # Volcado incremental
    pd.DataFrame.from_dict(best_cfg,orient='index').reset_index()\
       .rename(columns={'index':'subject'})\
       .to_csv(OUTPUT/"best_cfg_ensemble_v4.csv",index=False)

    # ── NUEVO: gráficos/CSV por sujeto + agregar a globales
    if best_out_dir is not None and best_cm is not None:
        cm_path_png = best_out_dir / f"{subj}_confusion_with_rejection.png"
        cm_path_csv = best_out_dir / f"{subj}_confusion_with_rejection.csv"
        plot_confusion_png(best_cm, f"{subj} · Matriz de confusión (con rechazo)", cm_path_png)
        pd.DataFrame(best_cm, index=[LABEL_NAMES[1], LABEL_NAMES[2], LABEL_NAMES[0]],
                               columns=[LABEL_NAMES[1], LABEL_NAMES[2], LABEL_NAMES[0]]).to_csv(cm_path_csv)
        # Violin por sujeto
        vio_path_png = best_out_dir / f"{subj}_violin_features.png"
        violin_features_png(best_X7_all, best_y_all, FEATURE_LABELS, f"{subj} · Distribución de características", vio_path_png)
        # Agregados globales
        global_cm += best_cm
        global_X7_list.append(best_X7_all)
        global_y_list.append(best_y_all)

# ── NUEVO: Gráficos globales al final ─────────────────────────────────────────
try:
    # 1) Matriz de confusión GLOBAL (sumando sujetos)
    cm_global_png = OUTPUT / "GLOBAL_confusion_with_rejection.png"
    plot_confusion_png(global_cm, "Global · Matriz de confusión (con rechazo)", cm_global_png)
    pd.DataFrame(global_cm, index=[LABEL_NAMES[1], LABEL_NAMES[2], LABEL_NAMES[0]],
                           columns=[LABEL_NAMES[1], LABEL_NAMES[2], LABEL_NAMES[0]]).to_csv(OUTPUT / "GLOBAL_confusion_with_rejection.csv")

    # 2) Violin GLOBAL de las 7 características (stack de todos los sujetos)
    if len(global_X7_list) > 0:
        X7_global = np.vstack(global_X7_list)
        y_global  = np.concatenate(global_y_list)
        violin_features_png(X7_global, y_global, FEATURE_LABELS, "Global · Distribución de características", OUTPUT / "GLOBAL_violin_features.png")

    # 3) Panel de desempeño por sujeto (barras + línea de rechazo)
    df_global = pd.DataFrame.from_dict(best_cfg, orient='index').reset_index().rename(columns={'index': 'subject'})
    perf_png = OUTPUT / "GLOBAL_performance_panel.png"
    if {"subject","acc_holdout","accuracy_no_rejection","rej_rate"}.issubset(df_global.columns):
        plot_global_performance(df_global[["subject","acc_holdout","accuracy_no_rejection","rej_rate"]], perf_png)
except Exception as e:
    warnings.warn(f"No se pudieron generar gráficos globales: {e}")

# Resumen global simple
print("\nPipeline V4 (adaptativo + rechazo + gráficas) completado.")
