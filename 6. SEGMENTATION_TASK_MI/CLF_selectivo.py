# train_ensemble_v5_selective.py
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#  BCI-fNIRS · Comité + Stacking (V5 selectivo + rechazo + calibración + PKL robusto)
#
#  (A) Guarda meta_input/use_logit_meta/experts_kept(class_order) en PKL
#  (B) Clases custom movidas a bci_custom_estimators.py → pickle estable
#  (C) Best split por utilidad selectiva (BA_aceptados − |rej−target| − λ·ECE)
#
#  Salidas por sujeto:
#   - confusion_with_rejection (3x3)
#   - confusion_no_rejection  (2x2)
#   - risk_coverage_curve
#   - calibration (reliability diagram)
#   - violin 7-features
#
#  Salidas globales:
#   - GLOBAL_confusion_with_rejection (3x3)
#   - GLOBAL_confusion_no_rejection  (2x2)
#   - GLOBAL_risk_coverage_mean
#   - GLOBAL_calibration
#   - GLOBAL_violin_features
#   - GLOBAL_performance_panel
# ──────────────────────────────────────────────────────────────────────────────

import os
from joblib.externals.loky import set_loky_pickler

# ---------- CONTROL DE HILOS GLOBALES ----------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
set_loky_pickler("pickle")
N_JOBS = 4

import numpy as np
import scipy.io as sio
import pandas as pd
import warnings
import pickle
import statistics
import random
from pathlib import Path
from collections import defaultdict
from joblib import parallel_backend

from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_predict,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFECV, RFE, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

# IMPORTANTE: clases custom en módulo (B)
from bci_custom_estimators import XGBSafeBinary, SafeBoruta, SafeSelectFromModel

# --- Plotting (backend no interactivo) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════════════════
SEED = 0
np.random.seed(SEED)
random.seed(SEED)

DATA_DIR  = Path("SEGMENTATION_TASK_MI")
OUTPUT    = Path("TRAINED_MODEL_IMPROVED_V5_SELECTIVE")
OUTPUT.mkdir(exist_ok=True, parents=True)
SUBJECTS  = [f"subject {i:02}" for i in range(1, 29)]

N_FOLDS, N_SPLITS, TEST_SIZE = 5, 15, 0.25
K_MIN, MIN_HEMI              = 6, 3
R2_CANDS                     = [0.05, 0.04, 0.03]

MIN_FEAT, MAX_FEAT = 6, 10

# Adaptación de rechazo y poda
TARGET_REJ_BASE = 0.10
REJ_RANGE = (0.05, 0.18)               # bounds razonables
TARGET_GRID = [0.08, 0.10, 0.12, 0.15]
PRUNE_WEAK_EXPERTS = True
WEAK_WEIGHT_THR = 0.08
WEAK_BA_THR = 0.55
USE_LOGIT_META = True                  # meta_input = logit_pairs

# Pesos utilidades (C)
LAMBDA_REJ_THR = 1.0                   # en búsqueda de THR (OOF)
LAMBDA_ECE_THR = 0.25

LAMBDA_REJ_SPLIT = 1.0                 # para escoger best split (holdout)
LAMBDA_ECE_SPLIT = 0.25

# Ventanas
CATALOG = {
    "A": (-5, 0),  "B": (0, 10),  "C": (5, 8),   "D": (10, 13),
    "E": (15, 18), "F": (16, 19), "G": (-5, -2), "H": (17, 20),
    "I": (2, 12),  "J": (3, 13),  "K": (4, 14),  "L": (5, 15)
}
HEMISPHERE = {'L': np.arange(0, 12), 'R': np.arange(12, 24)}

# Etiquetas
LABEL_NAMES = {1: "Mano derecha", 2: "Mano izquierda", 0: "Rechazo"}
CLASS_ORDER = [1, 2]  # para todo el pipeline


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def r2_signed(epo, y):
    cls = [epo[y == k] for k in (1, 2)]
    mu  = [c.mean(0) for c in cls]
    var = [c.var(0, ddof=1) for c in cls]
    r2  = (mu[0] - mu[1])**2 / (var[0] + var[1] + 1e-12)
    return r2 * np.where(mu[0] > mu[1], 1, -1)

def slope(x):
    return np.diff(x, axis=1).mean(1)

def features(O, R, t_mask, ch_idx):
    """
    O, R: [N, T, C]
    t_mask: bool [T]
    ch_idx: índices de canales (0-based)
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
    F = np.stack(f, axis=2)  # [N, C, 7]
    return F.reshape(O_.shape[0], -1)

FEAT_PER_CH = 7

def cols_from_channels(ch_sel):
    return np.concatenate([np.arange(c * FEAT_PER_CH, (c + 1) * FEAT_PER_CH) for c in ch_sel])

FEATURE_LABELS = ["O_mean", "O_slope", "R_mean", "R_slope", "O_peak", "O_var", "O_minus_R_mean"]

def make_feature_names_for_cols(ch_sel):
    return [f"ch{int(c)}:{lab}" for c in ch_sel for lab in FEATURE_LABELS]


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


# ---------- Calibration / Selective Metrics ----------
def ece_from_confidence(y_true, proba, n_bins=15):
    """
    ECE clásico usando confidence=max(p) y exactitud por bin.
    y_true ∈ {1,2}. proba shape [N,2] con columnas en orden CLASS_ORDER.
    """
    y_true = np.asarray(y_true)
    conf = np.max(proba, axis=1)
    y_hat = np.argmax(proba, axis=1)
    # map idx->label
    y_hat_lbl = np.array(CLASS_ORDER, dtype=int)[y_hat]
    correct = (y_hat_lbl == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        acc_bin = correct[m].mean()
        conf_bin = conf[m].mean()
        ece += (m.mean()) * abs(acc_bin - conf_bin)
    return float(ece)

def risk_coverage_curve(y_true, proba):
    """
    Risk-Coverage: ordena por confianza (desc), y calcula riesgo acumulado
    risk(k)=1-accuracy(top-k), coverage=k/N.
    """
    y_true = np.asarray(y_true)
    conf = np.max(proba, axis=1)
    y_hat = np.argmax(proba, axis=1)
    y_hat_lbl = np.array(CLASS_ORDER, dtype=int)[y_hat]
    correct = (y_hat_lbl == y_true).astype(float)

    order = np.argsort(-conf)
    correct_sorted = correct[order]
    n = len(correct_sorted)

    cov = np.arange(1, n + 1) / n
    cum_correct = np.cumsum(correct_sorted)
    acc_k = cum_correct / np.arange(1, n + 1)
    risk = 1.0 - acc_k
    return cov, risk

def aurc_from_curve(cov, risk):
    return float(np.trapz(risk, cov))


def probs_to_logit_pairs(Z, eps=1e-6):
    """
    Z: [N, 2*E] concatenado p(class1),p(class2) por experto.
    Retorna [N, E] con log(p2/p1) por experto.
    """
    Z = np.clip(Z, eps, 1 - eps)
    exps = Z.shape[1] // 2
    feats = []
    for i in range(exps):
        p1 = Z[:, 2 * i]
        p2 = Z[:, 2 * i + 1]
        feats.append(np.log(p2 / p1))
    return np.column_stack(feats)

def meta_expert_weights_from_OOF(Z, y, expert_names):
    n_exp = len(expert_names)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")
    lr.fit(Z, y)
    coef = np.asarray(lr.coef_).ravel()
    p = Z.shape[1]

    if p == 2 * n_exp:
        w = [np.sum(np.abs(coef[2 * i:2 * i + 2])) for i in range(n_exp)]
    else:
        block = max(1, p // n_exp)
        w = []
        for i in range(n_exp):
            start = i * block
            end = min(start + block, p)
            w.append(float(np.sum(np.abs(coef[start:end]))))

    w = np.asarray(w, dtype=float)
    if not np.isfinite(w).any() or w.sum() <= 0:
        w = np.ones(n_exp, dtype=float) / n_exp
    else:
        w = w / w.sum()

    return {expert_names[i]: float(w[i]) for i in range(n_exp)}

def expert_col_indices(names, total_cols):
    # 2 columnas por experto (p1,p2)
    idx_map = {}
    for i, nm in enumerate(names):
        a, b = 2 * i, 2 * i + 1
        if b < total_cols:
            idx_map[nm] = (a, b)
    return idx_map


# ---------- Threshold search (OOF) ----------
def find_best_threshold_selective(
    y_true,
    proba,
    thr_min=0.50,
    thr_max=0.95,
    n=61,
    target_rej=0.10,
    rej_bounds=(0.05, 0.18),
    lambda_rej=1.0,
    lambda_ece=0.25,
    n_bins_ece=15
):
    """
    Maximiza: BA_aceptados - lambda_rej*|rej-target| - lambda_ece*ECE_aceptados
    sobre umbrales en [thr_min, thr_max].
    """
    y_true = np.asarray(y_true)
    conf = np.max(proba, axis=1)
    y_hat = np.argmax(proba, axis=1)
    y_hat_lbl = np.array(CLASS_ORDER, dtype=int)[y_hat]

    thrs = np.linspace(thr_min, thr_max, n)
    best = (float(thr_min), -np.inf, 1.0, 0.0, 1.0)  # thr, util, rej, ba_acc, ece_acc

    for t in thrs:
        accepted = conf >= t
        if accepted.sum() == 0:
            continue

        rej = 1.0 - accepted.mean()
        lo, hi = rej_bounds
        if not (lo <= rej <= hi):
            continue

        ba_acc = balanced_accuracy_safe(y_true[accepted], y_hat_lbl[accepted])
        ece_acc = ece_from_confidence(y_true[accepted], proba[accepted], n_bins=n_bins_ece)

        util = ba_acc - lambda_rej * abs(rej - target_rej) - lambda_ece * ece_acc

        if util > best[1]:
            best = (float(t), float(util), float(rej), float(ba_acc), float(ece_acc))

    return best


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ═════════════════════════════════════════════════════════════════════════════
def plot_confusion_png(cm, labels, title, out_png):
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_risk_coverage_png(cov, risk, title, out_png):
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=160)
    ax.plot(cov, risk, marker="o", markersize=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Coverage (1 - Rejection)")
    ax.set_ylabel("Risk (1 - Accuracy)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_calibration_png(y_true, proba, title, out_png, n_bins=15):
    """
    Reliability diagram con bins por confidence.
    """
    y_true = np.asarray(y_true)
    conf = np.max(proba, axis=1)
    y_hat = np.argmax(proba, axis=1)
    y_hat_lbl = np.array(CLASS_ORDER, dtype=int)[y_hat]
    correct = (y_hat_lbl == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    accs = []
    confs = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        bin_centers.append((lo + hi) / 2)
        accs.append(correct[m].mean())
        confs.append(conf[m].mean())

    ece = ece_from_confidence(y_true, proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=160)
    ax.plot([0, 1], [0, 1], linestyle="--")
    if len(bin_centers) > 0:
        ax.plot(confs, accs, marker="o")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE={ece:.3f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return float(ece)

def collapse_features_7(X_sel, n_channels, feat_per_ch=FEAT_PER_CH):
    Xr = X_sel.reshape(X_sel.shape[0], n_channels, feat_per_ch)  # [N, C, 7]
    return Xr.mean(axis=1)  # [N, 7]

def violin_features_png(X7, y, feature_labels, title, out_png):
    y = np.asarray(y)
    data1 = [X7[y == 1, i] for i in range(X7.shape[1])]
    data2 = [X7[y == 2, i] for i in range(X7.shape[1])]
    pos1 = np.arange(1, X7.shape[1] + 1) - 0.15
    pos2 = np.arange(1, X7.shape[1] + 1) + 0.15

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    ax.violinplot(data1, positions=pos1, widths=0.25, showmeans=True, showextrema=False, showmedians=False)
    ax.violinplot(data2, positions=pos2, widths=0.25, showmeans=True, showextrema=False, showmedians=False)
    ax.set_xticks(np.arange(1, X7.shape[1] + 1))
    ax.set_xticklabels(feature_labels, rotation=30, ha="right")
    ax.set_xlim(0.5, X7.shape[1] + 0.5)
    ax.set_title(title)
    ax.set_ylabel("Valor de la característica")
    ax.plot([], [], label=LABEL_NAMES[1])
    ax.plot([], [], label=LABEL_NAMES[2])
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_global_performance(df, out_png):
    d = df.copy()
    d["subject"] = d["subject"].astype(str)
    d = d.sort_values("util_holdout", ascending=False)

    x = np.arange(len(d))
    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=160)

    ax1.bar(x - 0.2, d["acc_holdout"].values, width=0.4, label="Acc (sin rechazo)")
    ax1.bar(x + 0.2, d["acc_accepted"].values, width=0.4, label="Acc (aceptados)")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(d["subject"].values, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, d["rej_rate"].values, marker="o", label="Rechazo")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Rechazo")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    ax1.set_title("Desempeño por sujeto (ordenado por utilidad selectiva)")
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# ACUMULADORES GLOBALES
# ═════════════════════════════════════════════════════════════════════════════
global_cm_3 = np.zeros((3, 3), dtype=int)  # labels [1,2,0]
global_cm_2 = np.zeros((2, 2), dtype=int)  # labels [1,2]
global_X7_list = []
global_y_list = []

global_cov_list = []
global_risk_list = []

global_calib_y = []
global_calib_p = []

best_cfg = {}  # por sujeto


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
for subj in SUBJECTS:
    print(f"\n{'='*70}\nProcesando {subj}\n{'='*70}")
    mat = DATA_DIR / subj / "Epochs_MI_classification.mat"
    if not mat.exists():
        warnings.warn(f"{subj}: .mat no encontrado - omitido")
        continue

    m = sio.loadmat(mat, squeeze_me=True)
    O, R = m["epochs_O"], m["epochs_R"]
    y = m["labels"].astype(int).ravel()
    N, T, C = O.shape

    # MI: 24 canales (0..23). Por eso siempre usamos np.arange(24).
    if C < 24:
        warnings.warn(f"{subj}: se esperaban 24 canales, pero hay {C}. Ajusta np.arange(C).")

    print(f"  Muestras: {N} | T: {T} | Canales: {C}")

    t_sec = np.linspace(-10, 25, T)
    VALID_KEYS = {"B", "C", "D", "E", "F", "I", "J", "K", "L"}
    masks = {k: (t_sec >= a) & (t_sec <= b) for k, (a, b) in CATALOG.items() if k in VALID_KEYS}

    sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)

    # Mejor split por utilidad selectiva (C)
    best_util_export = -np.inf
    best_model = None

    # artefactos del mejor split
    best_out_dir = None
    best_cm3 = None
    best_cm2 = None
    best_cov = None
    best_risk = None
    best_calib_y = None
    best_calib_p = None
    best_X7_all = None
    best_y_all = None

    # métricas mejores
    best_acc_hold = np.nan
    best_acc_acc = np.nan
    best_ba_acc = np.nan
    best_ece_acc = np.nan
    best_rej = np.nan
    best_thr_used = np.nan
    best_aurc = np.nan
    best_cv_mean = np.nan
    best_cv_std = np.nan
    best_win_mode = None
    best_thr_r2 = None
    best_ch = None

    with parallel_backend("loky", n_jobs=N_JOBS, inner_max_num_threads=1):
        for split_id, (idx_tr, idx_te) in enumerate(sss.split(np.zeros(N), y), 1):
            O_tr, R_tr, y_tr = O[idx_tr], R[idx_tr], y[idx_tr]
            O_te, R_te, y_te = O[idx_te], R[idx_te], y[idx_te]

            print(f"\n  === Split {split_id}/{N_SPLITS} ===  Train={len(y_tr)} Test={len(y_te)}")

            # Features por ventana usando 24 canales (0..23)
            X_win_tr = {
                k: features(O_tr, R_tr, tmask, np.arange(24)).astype(np.float32, copy=False)
                for k, tmask in masks.items()
            }
            X_win_te = {
                k: features(O_te, R_te, tmask, np.arange(24)).astype(np.float32, copy=False)
                for k, tmask in masks.items()
            }

            # ── Nested-CV ventana/canales ───────────────────────────────
            cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
            cfg_fold = defaultdict(list)
            acc_outer = []

            for tr_idx, va_idx in cv.split(np.zeros(len(y_tr)), y_tr):
                best_acc_fold = -1.0

                for k, t_mask in masks.items():
                    r2_tr = r2_signed(O_tr[tr_idx], y_tr[tr_idx])
                    abs_r2 = np.abs(r2_tr[t_mask].mean(0))
                    idx_sorted = np.argsort(-abs_r2)

                    for thr in R2_CANDS:
                        ch_sel = np.where(abs_r2 >= thr)[0]

                        # mínimo K_MIN
                        if ch_sel.size < K_MIN:
                            extra = [i for i in idx_sorted if i not in ch_sel][:K_MIN - ch_sel.size]
                            ch_sel = np.hstack([ch_sel, extra])

                        # hemisferios balanceados
                        for side in ("L", "R"):
                            hemi = HEMISPHERE[side]
                            present = np.intersect1d(ch_sel, hemi)
                            if present.size < MIN_HEMI:
                                extra = [i for i in idx_sorted if i in hemi and i not in ch_sel][:MIN_HEMI - present.size]
                                ch_sel = np.hstack([ch_sel, extra])

                        cols = cols_from_channels(ch_sel)
                        X_tr_ = X_win_tr[k][tr_idx][:, cols]
                        X_va_ = X_win_tr[k][va_idx][:, cols]

                        scaler = StandardScaler().fit(X_tr_)
                        X_tr_s = scaler.transform(X_tr_)
                        X_va_s = scaler.transform(X_va_)

                        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.3)
                        sel = RFECV(
                            lda, step=0.1, cv=3, n_jobs=1,
                            min_features_to_select=MIN_FEAT,
                            scoring="balanced_accuracy",
                        ).fit(X_tr_s, y_tr[tr_idx])

                        f_idx = sel.get_support(indices=True)

                        try:
                            bag = BalancedBaggingClassifier(
                                estimator=lda,
                                n_estimators=30,
                                max_samples=0.6,
                                max_features=0.8,
                                bootstrap=True,
                                bootstrap_features=True,
                                n_jobs=1,
                                sampling_strategy="auto",
                                random_state=SEED,
                            )
                            bag.fit(X_tr_s[:, f_idx], y_tr[tr_idx])
                            y_va_hat = bag.predict(X_va_s[:, f_idx])
                            acc = balanced_accuracy_safe(y_tr[va_idx], y_va_hat)
                        except Exception:
                            acc = 0.0

                        if acc > best_acc_fold:
                            best_acc_fold = acc
                            best_key = k
                            best_thr = thr
                            best_ch_local = ch_sel.copy()

                cfg_fold["win_key"].append(best_key)
                cfg_fold["thr"].append(best_thr)
                cfg_fold["ch"].append(best_ch_local)
                acc_outer.append(best_acc_fold)

            cv_mean_split = float(np.mean(acc_outer))
            cv_std_split = float(np.std(acc_outer))

            # ── Votación canales ────────────────────────────────────────
            win_mode = statistics.mode(cfg_fold["win_key"])
            thr_mode = statistics.mode(cfg_fold["thr"])

            ch_votes = np.concatenate(cfg_fold["ch"])
            uniq, cnt = np.unique(ch_votes, return_counts=True)
            ch_sel = uniq[cnt >= len(cfg_fold["ch"]) / 2]

            if ch_sel.size < K_MIN:
                extra = [u for u in uniq[np.argsort(-cnt)] if u not in ch_sel][:K_MIN - ch_sel.size]
                ch_sel = np.hstack([ch_sel, extra])

            cols_final = cols_from_channels(ch_sel)

            X_feat_tr = X_win_tr[win_mode][:, cols_final].astype(np.float32, copy=False)
            X_feat_te = X_win_te[win_mode][:, cols_final].astype(np.float32, copy=False)

            feat_names = make_feature_names_for_cols(ch_sel)
            print(f"    Ventana: {win_mode} {CATALOG[win_mode]} | Canales: {len(ch_sel)} | CV={cv_mean_split:.3f}±{cv_std_split:.3f}")

            # ── Expertos ───────────────────────────────────────────────
            lda_base = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.3)

            pipe_bag = Pipeline([
                ("scale", StandardScaler()),
                ("fs", RFECV(
                    lda_base, step=0.1, cv=5, n_jobs=1,
                    min_features_to_select=MIN_FEAT, scoring="balanced_accuracy"
                )),
                ("bag", BalancedBaggingClassifier(
                    estimator=lda_base, n_estimators=30,
                    max_samples=0.6, max_features=0.8,
                    bootstrap=True, bootstrap_features=True,
                    sampling_strategy="auto",
                    n_jobs=N_JOBS,
                    random_state=SEED
                ))
            ])

            pipe_svm = Pipeline([
                ("scale", StandardScaler()),
                ("fs", RFE(LinearSVC(C=0.1, penalty="l2", dual=False), n_features_to_select=MAX_FEAT, step=0.2)),
                ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced")),
            ])

            pipe_xgb = Pipeline([
                ("fs", SafeSelectFromModel(
                    XGBSafeBinary(
                        n_estimators=60, max_depth=3, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, random_state=SEED
                    ),
                    min_features=MIN_FEAT, threshold="median",
                    importance_getter="feature_importances_"
                )),
                ("xgb", XGBSafeBinary(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=3, gamma=0.1,
                    reg_alpha=0.1, reg_lambda=1,
                    random_state=SEED
                )),
            ])

            rf_base = RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                max_depth=6, min_samples_split=5, min_samples_leaf=3,
                max_features="sqrt", random_state=SEED, n_jobs=N_JOBS
            )
            pipe_rf = Pipeline([
                ("boruta", SafeBoruta(
                    estimator=RandomForestClassifier(n_estimators=50, random_state=SEED),
                    n_estimators="auto", verbose=0, random_state=SEED
                )),
                ("rf", rf_base),
            ])

            pipe_mlp = Pipeline([
                ("scale", StandardScaler()),
                ("fs", SelectKBest(f_classif, k=MAX_FEAT)),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(20,),
                    solver="adam",
                    learning_rate_init=1e-3,
                    alpha=0.001,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=15,
                    max_iter=2000,
                    random_state=SEED
                )),
            ])

            experts_tab = {
                "bag": pipe_bag,
                "svm": pipe_svm,
                "xgb": pipe_xgb,
                "rf":  pipe_rf,
                "mlp": pipe_mlp,
            }

            # ══════════════════════════════════════════════════════════════
            # OOF por experto + stacking + umbral adaptativo (selectivo)
            # ══════════════════════════════════════════════════════════════
            cv_meta = StratifiedKFold(5, shuffle=True, random_state=SEED)
            Z_tr_list, names, trained_models = [], [], []
            expert_ba = {}

            for name, pipe in experts_tab.items():
                try:
                    oof_proba = cross_val_predict(
                        pipe, X_feat_tr, y_tr, cv=cv_meta,
                        method="predict_proba", n_jobs=1
                    )
                    Z_tr_list.append(oof_proba)
                    names.append(name)

                    y_hat_i = np.array(CLASS_ORDER)[np.argmax(oof_proba, axis=1)]
                    expert_ba[name] = balanced_accuracy_safe(y_tr, y_hat_i)

                    pipe.fit(X_feat_tr, y_tr)
                    trained_models.append((name, pipe))
                except Exception as e:
                    print(f"    {name} fuera del stacking por error: {e}")

            if len(Z_tr_list) == 0:
                warnings.warn(f"{subj}: split {split_id} sin expertos válidos.")
                continue

            Z_tr = np.concatenate(Z_tr_list, axis=1)

            # meta_input
            Z_tr_meta = probs_to_logit_pairs(Z_tr) if USE_LOGIT_META else Z_tr
            meta_input = "logit_pairs" if USE_LOGIT_META else "proba_pairs"

            meta_weights = meta_expert_weights_from_OOF(Z_tr_meta, y_tr, names)

            # ── Poda ────────────────────────────────────────────────────
            keep = []
            for nm in names:
                w = meta_weights.get(nm, 0.0)
                ba_i = expert_ba.get(nm, 0.5)
                if (w >= WEAK_WEIGHT_THR) or (ba_i >= WEAK_BA_THR):
                    keep.append(nm)

            if PRUNE_WEAK_EXPERTS and len(keep) >= 2 and len(keep) < len(names):
                print(f"    Poda de expertos → {keep}")
                idx_map = expert_col_indices(names, Z_tr.shape[1])
                cols_keep = []
                for nm in keep:
                    a, b = idx_map[nm]
                    cols_keep.extend([a, b])

                Z_tr = Z_tr[:, cols_keep]
                Z_tr_meta = probs_to_logit_pairs(Z_tr) if USE_LOGIT_META else Z_tr
                meta_weights = meta_expert_weights_from_OOF(Z_tr_meta, y_tr, keep)

                names = keep[:]  # orden final
                trained_models = [(nm, est) for (nm, est) in trained_models if nm in keep]
            else:
                names = names[:]  # orden final

            experts_kept = names[:]  # (A) orden exacto para reconstruir Z_te

            # Meta OOF para umbral
            meta_oof = CalibratedClassifierCV(
                estimator=LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
                method="sigmoid",
                cv=5,
            )
            oof_meta_proba = cross_val_predict(
                meta_oof, Z_tr_meta, y_tr,
                method="predict_proba", cv=5, n_jobs=1
            )

            # Buscar THR selectivo en OOF
            conf_oof = np.max(oof_meta_proba, axis=1)
            qthr = float(np.quantile(conf_oof, TARGET_REJ_BASE))
            thr_lo = max(0.50, qthr - 0.07)
            thr_hi = min(0.95, qthr + 0.07)

            best_thr = None
            best_util = -np.inf
            best_rej = None
            best_ba = None
            best_ece = None

            for tgt in TARGET_GRID:
                thr, util, rej, ba_acc, ece_acc = find_best_threshold_selective(
                    y_tr, oof_meta_proba,
                    thr_min=thr_lo, thr_max=thr_hi, n=61,
                    target_rej=float(tgt),
                    rej_bounds=REJ_RANGE,
                    lambda_rej=LAMBDA_REJ_THR,
                    lambda_ece=LAMBDA_ECE_THR,
                    n_bins_ece=15
                )
                if util > best_util:
                    best_thr, best_util, best_rej, best_ba, best_ece = thr, util, rej, ba_acc, ece_acc

            if best_thr is None:
                thr, util, rej, ba_acc, ece_acc = find_best_threshold_selective(
                    y_tr, oof_meta_proba,
                    thr_min=0.50, thr_max=0.95, n=91,
                    target_rej=float(TARGET_REJ_BASE),
                    rej_bounds=REJ_RANGE,
                    lambda_rej=LAMBDA_REJ_THR,
                    lambda_ece=LAMBDA_ECE_THR,
                    n_bins_ece=15
                )
                best_thr, best_util, best_rej, best_ba, best_ece = thr, util, rej, ba_acc, ece_acc

            THR_REJECT = float(best_thr)
            print(f"    THR adaptativo: {THR_REJECT:.3f} | OOF rej≈{best_rej:.2%} | BA_acc={best_ba:.3f} | ECE_acc={best_ece:.3f}")

            # Entrenar meta final
            meta = CalibratedClassifierCV(
                estimator=LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
                method="sigmoid",
                cv=5,
            )
            meta.fit(Z_tr_meta, y_tr)

            # TEST: construir Z_te en orden experts_kept
            Z_te_list = []
            for nm, model in trained_models:
                # trained_models ya filtrado; pero preserva orden experts_kept
                pass
            nm_to_model = {nm: mdl for nm, mdl in trained_models}
            for nm in experts_kept:
                Z_te_list.append(nm_to_model[nm].predict_proba(X_feat_te))

            Z_te_full = np.concatenate(Z_te_list, axis=1)
            Z_te_meta = probs_to_logit_pairs(Z_te_full) if USE_LOGIT_META else Z_te_full

            proba_final = meta.predict_proba(Z_te_meta)  # columnas en orden CLASS_ORDER
            conf = np.max(proba_final, axis=1)
            y_pred = np.array(CLASS_ORDER)[np.argmax(proba_final, axis=1)]

            # recentrado suave para mantener rechazo cerca del OOF estimado
            thr_used = THR_REJECT
            if best_rej is not None:
                rej_now = float(np.mean(conf < thr_used))
                if abs(rej_now - float(best_rej)) > 0.08:
                    thr_used = float(np.quantile(conf, np.clip(best_rej, REJ_RANGE[0], REJ_RANGE[1])))

            accepted = conf >= thr_used
            y_pred_rej = y_pred.copy()
            y_pred_rej[~accepted] = 0

            rej_rate = float(np.mean(y_pred_rej == 0))
            acc_hold = float(accuracy_score(y_te, y_pred))  # sin rechazo (2-clases)
            acc_acc = float(accuracy_score(y_te[accepted], y_pred[accepted])) if accepted.any() else 0.0
            ba_acc = float(balanced_accuracy_safe(y_te[accepted], y_pred[accepted])) if accepted.any() else 0.0
            ece_acc = float(ece_from_confidence(y_te[accepted], proba_final[accepted], n_bins=15)) if accepted.any() else 1.0

            cov, risk = risk_coverage_curve(y_te, proba_final)
            aurc = aurc_from_curve(cov, risk)

            # UTILIDAD DEL SPLIT (C)
            target = float(best_rej) if best_rej is not None else float(TARGET_REJ_BASE)
            util_split = ba_acc - LAMBDA_REJ_SPLIT * abs(rej_rate - target) - LAMBDA_ECE_SPLIT * ece_acc

            print(f"    Holdout: acc={acc_hold:.3f} | acc_acc={acc_acc:.3f} | BA_acc={ba_acc:.3f} | rej={rej_rate:.2%} | ECE_acc={ece_acc:.3f} | util={util_split:.3f}")

            # Si es el mejor split, guarda todo
            if util_split > best_util_export:
                best_util_export = util_split
                best_acc_hold = acc_hold
                best_acc_acc = acc_acc
                best_ba_acc = ba_acc
                best_ece_acc = ece_acc
                best_rej = rej_rate
                best_thr_used = thr_used
                best_aurc = aurc
                best_cv_mean = cv_mean_split
                best_cv_std = cv_std_split
                best_win_mode = win_mode
                best_thr_r2 = thr_mode
                best_ch = ch_sel.copy()

                # confusion matrices
                cm3 = confusion_matrix(y_te, y_pred_rej, labels=[1, 2, 0])
                cm2 = confusion_matrix(y_te, y_pred, labels=[1, 2])

                best_cm3 = cm3.copy()
                best_cm2 = cm2.copy()
                best_cov = cov.copy()
                best_risk = risk.copy()
                best_calib_y = y_te[accepted].copy() if accepted.any() else np.array([], dtype=int)
                best_calib_p = proba_final[accepted].copy() if accepted.any() else np.zeros((0, 2), dtype=float)

                # Violin: todas las épocas del sujeto (ventana+canales del mejor split)
                X_all_win = features(O, R, masks[win_mode], np.arange(24)).astype(np.float32, copy=False)
                X_sel_all = X_all_win[:, cols_final]
                X7_all = collapse_features_7(X_sel_all, n_channels=len(ch_sel))
                best_X7_all = X7_all.copy()
                best_y_all = y.copy()

                # paquete del modelo (A)
                model_pkg = dict(
                    version="V5_selective",
                    seed=int(SEED),

                    models=trained_models,
                    stacking_meta=meta,

                    expert_names=experts_kept,     # orden exacto para armar Z_te
                    experts_kept=experts_kept,     # alias (compat)
                    class_order=CLASS_ORDER,       # para interpretar proba

                    use_logit_meta=bool(USE_LOGIT_META),
                    meta_input=meta_input,

                    t_mask=masks[win_mode],
                    n_samples_win=int(masks[win_mode].sum()),
                    channels=ch_sel.copy(),
                    win_key=win_mode,
                    win_sec=CATALOG[win_mode],

                    # métricas best split
                    acc_holdout=float(acc_hold),
                    acc_accepted=float(acc_acc),
                    ba_accepted=float(ba_acc),
                    ece_accepted=float(ece_acc),
                    rej_rate=float(rej_rate),
                    thr_reject=float(thr_used),
                    target_rej=float(target),
                    aurc=float(aurc),
                    util_holdout=float(util_split),

                    meta_weights=meta_weights,
                )

                out_dir = OUTPUT / subj
                out_dir.mkdir(exist_ok=True, parents=True)
                best_out_dir = out_dir

                with open(out_dir / f"{subj}_model_ensemble.pkl", "wb") as f:
                    pickle.dump(model_pkg, f, protocol=pickle.HIGHEST_PROTOCOL)

                # guarda holdout para reproducir métricas/plots
                sio.savemat(out_dir / f"{subj}_holdout25.mat", {"O": O_te, "R": R_te, "labels": y_te})

    # Si no encontró split válido
    if best_model is None and best_out_dir is None:
        if not np.isfinite(best_util_export):
            warnings.warn(f"{subj}: no se pudo fijar best split — omitido.")
            continue

    # ── Guardar plots por sujeto del mejor split ─────────────────────────────
    if best_out_dir is not None:
        # 3x3 (con rechazo)
        plot_confusion_png(
            best_cm3,
            labels=[LABEL_NAMES[1], LABEL_NAMES[2], LABEL_NAMES[0]],
            title=f"{subj} · Confusión (con rechazo)",
            out_png=best_out_dir / f"{subj}_confusion_with_rejection.png",
        )
        pd.DataFrame(
            best_cm3,
            index=[LABEL_NAMES[1], LABEL_NAMES[2], LABEL_NAMES[0]],
            columns=[LABEL_NAMES[1], LABEL_NAMES[2], LABEL_NAMES[0]],
        ).to_csv(best_out_dir / f"{subj}_confusion_with_rejection.csv")

        # 2x2 (sin rechazo)
        plot_confusion_png(
            best_cm2,
            labels=[LABEL_NAMES[1], LABEL_NAMES[2]],
            title=f"{subj} · Confusión (sin rechazo)",
            out_png=best_out_dir / f"{subj}_confusion_no_rejection.png",
        )
        pd.DataFrame(
            best_cm2,
            index=[LABEL_NAMES[1], LABEL_NAMES[2]],
            columns=[LABEL_NAMES[1], LABEL_NAMES[2]],
        ).to_csv(best_out_dir / f"{subj}_confusion_no_rejection.csv")

        # Risk-Coverage
        plot_risk_coverage_png(
            best_cov, best_risk,
            title=f"{subj} · Risk–Coverage (AURC={best_aurc:.3f})",
            out_png=best_out_dir / f"{subj}_risk_coverage.png",
        )

        # Calibration (solo aceptados)
        if best_calib_y.size > 0:
            plot_calibration_png(
                best_calib_y, best_calib_p,
                title=f"{subj} · Calibración (aceptados)",
                out_png=best_out_dir / f"{subj}_calibration_accepted.png",
                n_bins=15,
            )

        # Violin features
        if best_X7_all is not None:
            violin_features_png(
                best_X7_all, best_y_all,
                FEATURE_LABELS,
                title=f"{subj} · Distribución de características (ventana {best_win_mode})",
                out_png=best_out_dir / f"{subj}_violin_features.png",
            )

        # Agregados globales
        global_cm_3 += best_cm3
        global_cm_2 += best_cm2
        if best_X7_all is not None:
            global_X7_list.append(best_X7_all)
            global_y_list.append(best_y_all)
        if best_cov is not None:
            global_cov_list.append(best_cov)
            global_risk_list.append(best_risk)
        if best_calib_y is not None and best_calib_y.size > 0:
            global_calib_y.append(best_calib_y)
            global_calib_p.append(best_calib_p)

        # Registro CSV por sujeto
        row = dict(
            subject=subj,
            win_key=best_win_mode,
            win_sec_start=CATALOG[best_win_mode][0],
            win_sec_end=CATALOG[best_win_mode][1],
            r2_thr=best_thr_r2,
            n_channels=int(len(best_ch)) if best_ch is not None else np.nan,
            channels="|".join(map(str, best_ch.tolist())) if best_ch is not None else "",
            CV_mean=float(best_cv_mean),
            CV_std=float(best_cv_std),

            acc_holdout=float(best_acc_hold),
            acc_accepted=float(best_acc_acc),
            ba_accepted=float(best_ba_acc),
            ece_accepted=float(best_ece_acc),

            rej_rate=float(best_rej),
            thr_reject=float(best_thr_used),
            aurc=float(best_aurc),
            util_holdout=float(best_util_export),
            meta_input=("logit_pairs" if USE_LOGIT_META else "proba_pairs"),
            use_logit_meta=bool(USE_LOGIT_META),
        )

        best_cfg[subj] = row
        pd.DataFrame(list(best_cfg.values())).to_csv(OUTPUT / "best_cfg_ensemble_v5.csv", index=False)

        print(f"\n{subj} BEST:")
        print(f"  util={best_util_export:.3f} | acc={best_acc_hold:.3f} | acc_acc={best_acc_acc:.3f} | BA_acc={best_ba_acc:.3f}")
        print(f"  rej={best_rej:.2%} | ECE_acc={best_ece_acc:.3f} | AURC={best_aurc:.3f} | THR={best_thr_used:.3f}")

# ── GLOBAL PLOTS ─────────────────────────────────────────────────────────────
try:
    # Confusión global
    plot_confusion_png(
        global_cm_3,
        labels=[LABEL_NAMES[1], LABEL_NAMES[2], LABEL_NAMES[0]],
        title="Global · Confusión (con rechazo)",
        out_png=OUTPUT / "GLOBAL_confusion_with_rejection.png",
    )
    pd.DataFrame(
        global_cm_3,
        index=[LABEL_NAMES[1], LABEL_NAMES[2], LABEL_NAMES[0]],
        columns=[LABEL_NAMES[1], LABEL_NAMES[2], LABEL_NAMES[0]],
    ).to_csv(OUTPUT / "GLOBAL_confusion_with_rejection.csv")

    plot_confusion_png(
        global_cm_2,
        labels=[LABEL_NAMES[1], LABEL_NAMES[2]],
        title="Global · Confusión (sin rechazo)",
        out_png=OUTPUT / "GLOBAL_confusion_no_rejection.png",
    )
    pd.DataFrame(
        global_cm_2,
        index=[LABEL_NAMES[1], LABEL_NAMES[2]],
        columns=[LABEL_NAMES[1], LABEL_NAMES[2]],
    ).to_csv(OUTPUT / "GLOBAL_confusion_no_rejection.csv")

    # Violin global
    if len(global_X7_list) > 0:
        X7_global = np.vstack(global_X7_list)
        y_global = np.concatenate(global_y_list)
        violin_features_png(
            X7_global, y_global,
            FEATURE_LABELS,
            title="Global · Distribución de características",
            out_png=OUTPUT / "GLOBAL_violin_features.png",
        )

    # Risk-Coverage promedio (interpolando a grid común)
    if len(global_cov_list) > 0:
        grid = np.linspace(0.0, 1.0, 50)
        risks_interp = []
        for cov, risk in zip(global_cov_list, global_risk_list):
            risks_interp.append(np.interp(grid, cov, risk))
        mean_risk = np.mean(np.vstack(risks_interp), axis=0)

        plot_risk_coverage_png(
            grid, mean_risk,
            title="Global · Risk–Coverage (promedio)",
            out_png=OUTPUT / "GLOBAL_risk_coverage_mean.png",
        )

    # Calibración global (aceptados)
    if len(global_calib_y) > 0:
        y_cal = np.concatenate(global_calib_y)
        p_cal = np.vstack(global_calib_p)
        plot_calibration_png(
            y_cal, p_cal,
            title="Global · Calibración (aceptados)",
            out_png=OUTPUT / "GLOBAL_calibration_accepted.png",
            n_bins=15,
        )

    # Panel desempeño global (por sujeto)
    df = pd.read_csv(OUTPUT / "best_cfg_ensemble_v5.csv")
    if {"subject", "acc_holdout", "acc_accepted", "rej_rate", "util_holdout"}.issubset(df.columns):
        plot_global_performance(
            df[["subject", "acc_holdout", "acc_accepted", "rej_rate", "util_holdout"]],
            OUTPUT / "GLOBAL_performance_panel.png",
        )

except Exception as e:
    warnings.warn(f"No se pudieron generar gráficos globales: {e}")

print("\nPipeline V5 (selectivo + rechazo + calibración + PKL robusto) completado.")
