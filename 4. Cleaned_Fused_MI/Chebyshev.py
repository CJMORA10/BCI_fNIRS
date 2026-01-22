import os, numpy as np, scipy.io as sio
from scipy.signal import cheby2, sosfiltfilt          # ← cheby2 + SOS
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────── Parámetros globales ───────────
FS       = 10                 # Hz  (misma fs del dataset)
BAND     = (0.01, 0.09)       # banda pasa‑banda en Hz
ORDER    = 3                  # orden del filtro
ATT_SB   = 30                 # atenuación stop‑band (dB) → igual que demo BBCI
padlen   = 3*(ORDER+1)        # mínimo requerido por filtfilt / sosfiltfilt

CLEAN_DIR = Path("Cleaned_Fused_MI")
OUT_DIR   = Path("Filtered_Cheby_Cleaned_Fused_MI")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────── Filtro pasa‑banda Chebyshev‑II ────────────
def cheby2_bandpass(data, low, high, fs, order=3, att_sb=30):
    nyq  = 0.5*fs
    wp   = [low/nyq,  high/nyq]          # banda de paso normalizada
    ws   = [0.7*wp[0], 1.3*wp[1]]        # banda de stop un poco más ancha
    sos  = cheby2(order, att_sb, ws, btype='bandpass', output='sos')
    # 0‑fase y estable numéricamente
    return sosfiltfilt(sos, data, axis=0, padlen=padlen)

# ─────────── Bucle por sujetos ────────────
subjects = [f"subject {i:02}" for i in range(1, 29)]

for subj in subjects:
    fpath = os.path.join(CLEAN_DIR, subj, "Cleaned_Fused_MI_MBLL_cnt.mat")
    if not os.path.isfile(fpath):
        print(f"{subj}: fichero no encontrado, omito.")
        continue

    mat  = sio.loadmat(fpath)
    HbO  = mat["HbO"].astype(float)      # (T × nCh)
    HbR  = mat["HbR"].astype(float)

    if HbO.shape[0] <= padlen:
        print(f"{subj}: señal demasiado corta para filtrar.")
        continue

    # 1) Filtrado Chebyshev‑II
    HbO_f = cheby2_bandpass(HbO, *BAND, FS, ORDER, ATT_SB)
    HbR_f = cheby2_bandpass(HbR, *BAND, FS, ORDER, ATT_SB)

    # 2) Guardar .mat filtrado
    out_dir = os.path.join(OUT_DIR, subj);  os.makedirs(out_dir, exist_ok=True)
    sio.savemat(os.path.join(out_dir, "Filtered_Fused_MI_MBLL_cnt.mat"),
                {"HbO": HbO_f, "HbR": HbR_f, "fs": FS,
                 "band": np.asarray(BAND), "order": ORDER,
                 "design": "cheby2", "att_sb": ATT_SB})

    # 3) Diagnóstico RMS
    rms_in  = np.sqrt((HbO**2).mean())
    rms_out = np.sqrt((HbO_f**2).mean())
    print(f"{subj}: filtrado OK  |  RMS {rms_in:.3e} → {rms_out:.3e}")

    # 4) Gráfica rápida
    # Graficar las señales filtradas para este segmento
    plt.figure(figsize=(15, 10))
    for ch in range(HbO_f.shape[1]):  # Iterar sobre los 48 columnas reindexados de los 24 canales (13 al 36)
        original_ch_idx = 13 + ch  # Canales motores del 13 al 36 fisiológicos
        plt.plot(HbO_f[:, ch], label=f"HbO-Oxy Ch {original_ch_idx}", linestyle='-', color='red',  alpha=.35, lw=.6)
        plt.plot(HbR_f[:, ch], label=f"HbR-Deoxy Ch {original_ch_idx}", linestyle='--', color='blue', alpha=.35, lw=.6)
    plt.title(f"{subj} - MI fusionada limpia y filtrada")
    plt.xlabel("muestras");  plt.ylabel("mmol/L")
    plt.legend(fontsize='small', ncol=2, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    # Guardar la gráfica en la carpeta del sujeto
    plt.savefig(os.path.join(out_dir, "Filtered_Cleaned_Fused_MI_plot.png"))
    plt.close()
    print(f"Filtered plot saved: {out_dir}")
