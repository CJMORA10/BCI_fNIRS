import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------------------------------
# 1.  Parámetros globales
# --------------------------------------------------------------------------
DATA_DIR  = Path("Dataset_fNIRS")                                           # raíz de los sujetos
OUT_DIR   = Path("MBLL_MI")
os.makedirs(OUT_DIR, exist_ok=True)
SUBJECTS  = [f"subject {i:02}" for i in range(1, 29)]                 # subject 01 … 28
MI_CELLS  = [0, 2, 4]                                                 # cnt{1,1 / 3 / 5}
FS_EXPECT = 10                                                        # Hz tras down‑sample
BASE_SEC  = 60                                                        # baseline de 60s

# 760 / 850 nm → ε (mm⁻¹ / (mmol·cm))  [λ_high, λ_low]
EPSILON_760_850 = np.array([
    [0.602, 1.486],   # HbO
    [1.798, 3.843]    # HbR
])

# DPF adulto (λ_high, λ_low) — las longitudes están invertidas respecto a ε
DPF_760_850 = (5.98, 7.15)   # 850 nm → 5.98, 760 nm → 7.15
DIST_CM     = 3.0

# --------------------------------------------------------------------------
# 2.  Función MBLL (réplica de proc_BeerLambert)
# --------------------------------------------------------------------------
def mbll(raw, *, extinction, dpf, dist_cm, baseline_slice, use_log10=True):
    """
    raw:          (t, 2·N) intensidades [λ_low, λ_high] por canal fisiológico
    extinction:   (2, 2) ε  [[HbO_λhigh, HbO_λlow],
                              [HbR_λhigh, HbR_λlow]]
    dpf:          (λ_high, λ_low)
    dist_cm:      distancia fuente‑detector en cm
    baseline_slice: slice con las muestras de reposo
    """
    raw = raw.astype(float) + 1e-10
    n_t, n_cols = raw.shape
    assert n_cols % 2 == 0, "Los canales deben venir por pares low-high."

    # ---------- 1. log‑ratio ----------
    baseline = raw[baseline_slice].mean(axis=0, keepdims=True)
    att = -np.log10(raw / baseline) if use_log10 else -np.log(raw / baseline)

    # ---------- 2. Matriz ε corregida por DPF y distancia ----------
    eps = extinction / 10.0                            # mmol · L⁻¹
    eps *= np.array(dpf)[:, None] * dist_cm            # (2,2)
    inv_eps = np.linalg.inv(eps)                       # (2,2)

    # ---------- 3. Resolver HbO/HbR canal por canal ----------
    n_pairs = n_cols // 2
    HbO = np.empty((n_t, n_pairs))
    HbR = np.empty_like(HbO)

    for p in range(n_pairs):
        a_low  = att[:, 2*p]       # λ_low
        a_high = att[:, 2*p + 1]   # λ_high
        C = inv_eps @ np.vstack([a_high, a_low])       # (2, t)
        HbO[:, p], HbR[:, p] = C

    return HbO, HbR

# --------------------------------------------------------------------------
# 3.  Bucle principal por sujeto
# --------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

for subj in SUBJECTS:
    subj_path = os.path.join(DATA_DIR, subj)
    cnt_file  = os.path.join(subj_path, "cnt.mat")
    if not os.path.isfile(cnt_file):
        print(f"[{subj}] cnt.mat no encontrado. Se omite.")
        continue

    out_subj = os.path.join(OUT_DIR, subj)
    os.makedirs(out_subj, exist_ok=True)

    cnt_cells = sio.loadmat(cnt_file, squeeze_me=True)["cnt"]

    mbll_cells = []

    for cell_idx in MI_CELLS:
        cell = cnt_cells[cell_idx]
        raw  = cell["x"].item() if isinstance(cell["x"], np.ndarray) and cell["x"].ndim==0 else cell["x"]  # (t, 72)
        fs   = int(cell["fs"].item())
        if fs != FS_EXPECT:
            print(f"[{subj}] Advertencia: fs={fs} Hz, se esperaba 10 Hz.")

        # -------- 3.a Selección y orden de canales motores --------
        # lowWL: índices 12–35 (C3/C4), highWL: +36
        motor_low  = range(12, 36)
        motor_cols = [i for p in motor_low for i in (p, p + 36)]
        motor_raw  = raw[:, motor_cols]               # (t, 48) low‑high‑…

        # -------- 3.b MBLL --------
        HbO, HbR = mbll(
            motor_raw,
            extinction=EPSILON_760_850,
            dpf=DPF_760_850,
            dist_cm=DIST_CM,
            baseline_slice=slice(None), #mejor resultado con baseline de todo el segmento en vez de 60s
            use_log10=True
        )

        mbll_cells.append({'HbO': HbO, 'HbR': HbR, 'fs': fs})

        # --------------------------------------------------------------------------
        # Fusión de las 3 sesiones MI  (equivalente a proc_appendCnt)
        # --------------------------------------------------------------------------
        # ‑ Después de construir la lista `cleaned_segments`

        # concatenar HbO y HbR de las 3 celdas  -------------------------------
        HbO_fused = np.concatenate([seg['HbO']  for seg in mbll_cells], axis=0)
        HbR_fused = np.concatenate([seg['HbR']  for seg in mbll_cells], axis=0)

        # vector con los límites de cada sesión dentro de la señal larga
        #      útil si luego re‑segmentar.
        session_len = [seg['HbO'].shape[0] for seg in mbll_cells]          # nº de muestras por sesión
        session_onsets = np.cumsum([0] + session_len[:-1])                       # índices inicio

        # guardar en un .mat adicional  --------------------------------------
        fused_file_path = os.path.join(out_subj, "Fused_MI_MBLL_cnt.mat")
        sio.savemat(
            fused_file_path,
            {
                'HbO': HbO_fused,          # 2‑D  (tiempo_total × 24 canales)
                'HbR': HbR_fused,
                'fs':  fs,                 # 10 Hz
                'session_onsets': session_onsets,   #  [0, len1, len1+len2]
                'session_len':     session_len      #  [len1, len2, len3]
            }
        )
        print(f"Archivo MI fusionado guardado: {fused_file_path}")

        # --------------------------------------------------------------------------
        # 3.e  —  Graficar la señal fusionada de MI
        # --------------------------------------------------------------------------
        # Carga de la fusión
        f = sio.loadmat(fused_file_path)
        HbO_f = f['HbO']        # (T_total, 24)
        HbR_f = f['HbR']        # (T_total, 24)
        onsets = f['session_onsets'].flatten()  # ej. [0, len1, len1+len2]
        fs = int(f['fs'].item())

        # --- Gráfico con eje en muestras y longitud exacta ---
        n_samples = HbO_f.shape[0]
        samples   = np.arange(n_samples)

        fig, ax = plt.subplots(figsize=(12, 6))
        for ch in range(HbO_f.shape[1]):
            ax.plot(samples, HbO_f[:, ch], linestyle='-',  color='red',   alpha=0.3, linewidth=0.8)
            ax.plot(samples, HbR_f[:, ch], linestyle='--', color='blue',  alpha=0.3, linewidth=0.8)

        # líneas verticales de separación de sesiones en muestras
        for onset in session_onsets:
            ax.axvline(onset, color='k', linestyle=':', linewidth=1)

        # anotar la longitud
        ax.text(0.99, 0.01, f'n = {n_samples} muestras',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5))

        ax.set_xlabel("Muestras")
        ax.set_ylabel("Concentración (mmol/L)")
        ax.set_title("Fusión de 3 sesiones MI - Índice de muestra")
        labels = (["HbO Ch " + str(13+ch) for ch in range(HbO_f.shape[1])] +
                ["HbR Ch " + str(13+ch) for ch in range(HbR_f.shape[1])])
        ax.legend(labels, ncol=2, fontsize='small',
                loc='upper left', bbox_to_anchor=(1.05, 1))

        plt.tight_layout()
        plt.savefig(os.path.join(out_subj, "Fused_MI_samples_exact_length.png"))
        plt.close()

        # -------- 3.c Gráfica rápida --------
        plt.figure(figsize=(12, 6))
        for ch in range(HbO.shape[1]):
            original_ch_idx = 13 + ch  # Canales motores del 13 al 36 fisiológicos
            plt.plot(HbO[:, ch], label=f"HbO-Oxy Ch {original_ch_idx}", linestyle='-', color='red',  alpha=.35, lw=.6)
            plt.plot(HbR[:, ch], label=f"HbR-Deoxy Ch {original_ch_idx}", linestyle='--', color='blue', alpha=.35, lw=.6)
        plt.title(f"{subj} - MI sesión {cell_idx+1}  (área motora)")
        plt.xlabel("Muestras");  plt.ylabel("mmol/L")
        plt.legend(fontsize='small', ncol=2, loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        
        plt.savefig(os.path.join(out_subj, f"MI_{cell_idx+1}_motor.png"))
        plt.close()

    # -------- 3.d Guardar .mat --------
    sio.savemat(os.path.join(out_subj, "MBLL_converted_cnt.mat"),
                {'cnt': mbll_cells})
    print(f"[{subj}] terminado y guardado.")