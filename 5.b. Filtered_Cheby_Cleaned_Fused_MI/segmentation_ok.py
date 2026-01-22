import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ----------------------------------------------------------
# 1. Parámetros globales de segmentación
# ----------------------------------------------------------
FS          = 10
IVAL_EPO    = (-10, 25)                 # seg  →  350 muestras
IVAL_BASE   = (-3, 0)                   # seg  →  para baseline
WIN_TASK    = (0, 10)                   # seg  →  100 muestras
WIN_REST    = (15, 25)                  # seg  →  100 muestras
MI_IDX      = [0, 2, 4]                 # mrk{1,1 /3 /5}

# ----------------------------------------------------------
# 2. Rutas base
# ----------------------------------------------------------
RAW_DIR     = Path("Dataset_fNIRS")                     # contiene cnt.mat y mrk.mat
FUSED_DIR   = Path("Filtered_Cheby_Cleaned_Fused_MI")                 # contiene Filtered_Fused_MI_MBLL_cnt.mat
OUT_DIR     = Path("SEGMENTATION_TASK_MI")       # salida

OUT_DIR.mkdir(exist_ok=True, parents=True)
subjects = [f"subject {i:02}" for i in range(1, 29)]    # 01 … 28

# ----------------------------------------------------------
# 3. Función de segmentación + graficado
# ----------------------------------------------------------
def segment_subject(subj):
    subj_raw   = RAW_DIR   / subj
    subj_fused = FUSED_DIR / subj
    subj_out   = OUT_DIR   / subj
    subj_out.mkdir(exist_ok=True, parents=True)

    fused_file = subj_fused / 'Filtered_Fused_MI_MBLL_cnt.mat'
    mrk_file   = subj_raw   / 'mrk.mat'
    cnt_file   = subj_raw   / 'cnt.mat'

    if not (fused_file.exists() and mrk_file.exists() and cnt_file.exists()):
        print(f"[{subj}]  archivos faltantes → omitido")
        return

    # --- Señal fusionada (HbO / HbR) ---
    fdat = sio.loadmat(fused_file)
    HbO, HbR = fdat['HbO'], fdat['HbR']
    Ttot     = HbO.shape[0]
    print(f"[{subj}] Ttot: {Ttot}, HbO shape: {HbO.shape}, HbR shape: {HbR.shape}")

    # --- Longitud de las 6 sesiones originales ---
    cnt_raw = sio.loadmat(cnt_file, simplify_cells=True)['cnt']  # ahora es lista
    len_all = [c['x'].shape[0] for c in cnt_raw]
    offset_all = np.cumsum([0] + len_all[:-1])
    print(f"[{subj}] len_all: {len_all}, offset_all: {offset_all}")

    # --- Marcadores LMI / RMI (sólo sesiones 1‑3‑5) ---
    mrk_raw = sio.loadmat(mrk_file, simplify_cells=True)['mrk']   # lista de 6 dicts
    onsets, labels = [], []

    # Longitudes de las tres sesiones MI
    mi_len = [len_all[i] for i in MI_IDX]            # [len1, len3, len5]

    # Desplazamiento acumulado dentro de la señal fusionada
    mi_offset = np.cumsum([0] + mi_len[:-1])         # [0, len1, len1+len3]

    for mi_block_idx, k in enumerate(MI_IDX):
        # 1)  mrk.time está en milisegundos  →  conviértelo a MUESTRAS
        t_rel = (mrk_raw[k]['time'] * FS / 1000).astype(int)   # FS = 10
        lab   = np.argmax(mrk_raw[k]['y'], 0) + 1              # 1 = LMI, 2 = RMI
        onsets.append(t_rel + mi_offset[mi_block_idx])         # 2 desplazamiento correcto
        labels.append(lab)
    onsets = np.hstack(onsets)
    labels = np.hstack(labels).astype(int)
    order  = np.argsort(onsets)
    onsets, labels = onsets[order], labels[order]
    print(f"[{subj}] adjusted onsets: {onsets}, labels: {labels}")
    print('min', onsets.min(), 'max', onsets.max(), 'Ttot', Ttot)
    # → ahora todos los onsets estarán entre 0 y ≈ 21 500
    print('Trials totales:', len(onsets))   # 60
    print(np.bincount(labels)[1:], '(LMI, RMI)')  # 30, 30

    # --- Segentación y baseline ---
    n_epo  = int((IVAL_EPO[1]-IVAL_EPO[0])*FS)   # 350
    shift  = int(IVAL_EPO[0]*FS)                 # -100
    b0 = int((IVAL_BASE[0]-IVAL_EPO[0])*FS)      # 70
    b1 = int((IVAL_BASE[1]-IVAL_EPO[0])*FS)      # 100
    print(f"[{subj}] n_epo: {n_epo}, shift: {shift}, b0: {b0}, b1: {b1}")

    epochs_O, epochs_R, lbl_good = [], [], []
    task_O, task_R, rest_O, rest_R = [], [], [], []

    i0_t, i1_t = [int((t-IVAL_EPO[0])*FS) for t in WIN_TASK]   # 100 muestras
    i0_r, i1_r = [int((t-IVAL_EPO[0])*FS) for t in WIN_REST]

    for idx, (t, y) in enumerate(zip(onsets, labels)):
        s, e = int(t + shift), int(t + shift + n_epo)  # Convert to integers
        print(f"[{subj}] Trial {idx + 1}: t={t}, s={s}, e={e}, Ttot={Ttot}")
        if s < 0 or e > Ttot:
            print(f"[{subj}] Skipping trial {idx + 1}: Out of bounds")
            continue

        ep_O = HbO[s:e].copy()
        ep_R = HbR[s:e].copy()
        ep_O -= ep_O[b0:b1].mean(axis=0, keepdims=True)
        ep_R -= ep_R[b0:b1].mean(axis=0, keepdims=True)

        # guardar en listas
        epochs_O.append(ep_O)
        epochs_R.append(ep_R)
        lbl_good.append(y)

        # sub‑ventanas tarea / reposo
        task_O.append(ep_O[i0_t:i1_t])
        task_R.append(ep_R[i0_t:i1_t])
        rest_O.append(ep_O[i0_r:i1_r])
        rest_R.append(ep_R[i0_r:i1_r])

        # --- graficar ---
        plot_segment(ep_O[i0_t:i1_t], ep_R[i0_t:i1_t],
                     ep_O[i0_r:i1_r], ep_R[i0_r:i1_r],
                     seg_idx=len(lbl_good), task_label=y,
                     out_dir=subj_out)

    # --- apilar y guardar ---
    if not epochs_O:
        print(f"[{subj}]  ningún ensayo válido")
        return
    epochs_O = np.stack(epochs_O)         # (30,350,24)
    epochs_R = np.stack(epochs_R)
    task_O   = np.stack(task_O)           # (30,100,24)
    task_R   = np.stack(task_R)
    rest_O   = np.stack(rest_O)
    rest_R   = np.stack(rest_R)
    labels   = np.array(lbl_good)

    sio.savemat(subj_out / 'Epochs_MI_classification.mat', {
        'epochs_O': epochs_O, 'epochs_R': epochs_R,
        'task_O': task_O,     'task_R': task_R,
        'rest_O': rest_O,     'rest_R': rest_R,
        'labels': labels,
        'fs': FS,
        'ival_epoch': IVAL_EPO,
        'ival_base':  IVAL_BASE
    })
    print(f"[{subj}]  épocas guardadas y gráficos generados")


# ----------------------------------------------------------
# 4. Función para graficar un ensayo (tarea y reposo)
# ----------------------------------------------------------
def plot_segment(task_O, task_R, rest_O, rest_R, seg_idx, task_label, out_dir: Path):
    task_str = 'LMI' if task_label == 1 else 'RMI'
    task_dir = out_dir / 'Task' / task_str
    rest_dir = out_dir / 'Rest'
    task_dir.mkdir(parents=True, exist_ok=True)
    rest_dir.mkdir(parents=True, exist_ok=True)

    # ----- tarea -----
    plt.figure(figsize=(15, 6))
    for ch in range(task_O.shape[1]):
        plt.plot(task_O[:, ch], label=f"HbO Ch {ch + 13}", color='red',  alpha=.35, lw=.7)
        plt.plot(task_R[:, ch], label=f"HbR Ch {ch + 13}", color='blue', alpha=.35, lw=.7, ls='--')
    plt.title(f'Sujeto {out_dir.name}  ·  Ensayo {seg_idx}  ({task_str})')
    plt.xlabel('Muestras  [0‑100 = 0‑10 s]')
    plt.ylabel('ΔHb  (mmol/L)')
    plt.legend(fontsize='small', ncol=2, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(task_dir / f'Seg{seg_idx:02}_{task_str}.png')
    plt.close()

    # ----- reposo -----
    plt.figure(figsize=(15, 6))
    for ch in range(rest_O.shape[1]):
        plt.plot(rest_O[:, ch], label=f"HbO Ch {ch + 13}", color='red',  alpha=.35, lw=.7)
        plt.plot(rest_R[:, ch], label=f"HbR Ch {ch + 13}", color='blue', alpha=.35, lw=.7, ls='--')
    plt.title(f'Sujeto {out_dir.name}  ·  Ensayo {seg_idx}  (Reposo)')
    plt.xlabel('Muestras  [0‑100 = 15‑25 s]')
    plt.ylabel('ΔHb  (mmol/L)')
    plt.legend(fontsize='small', ncol=2, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(rest_dir / f'Seg{seg_idx:02}_Rest.png')
    plt.close()

# ----------------------------------------------------------
# 5. Ejecutar para todos los sujetos
# ----------------------------------------------------------
for subj in subjects:
    segment_subject(subj)