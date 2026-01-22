import os, numpy as np, scipy.io as sio
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# ─────────── Parámetros globales ───────────
FS       = 10                # Hz
BAND     = (0.01, 0.09)      # Hz
ORDER    = 3                 # orden Butterworth
padlen   = 3*(ORDER+1)

CLEAN_DIR = r"C:/Users/carlo/OneDrive - correounivalle.edu.co/Escritorio/TG1/Cleaned_Fused_MI"      # <-- tu salida de artefactos
OUT_DIR   = r"C:/Users/carlo/OneDrive - correounivalle.edu.co/Escritorio/TG1/Filtered_Cleaned_Fused_MI"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────── Filtro paso‑banda ────────────
def butter_bandpass(data, low, high, fs, order=3):
    nyq   = 0.5*fs
    b, a  = butter(order, [low/nyq, high/nyq], btype='band')
    # filtfilt conserva fase 0 e introduce el retardo inverso
    return filtfilt(b, a, data, axis=0, padlen=padlen)

# ─────────── Bucle por sujetos ────────────
subjects = [f"subject {i:02}" for i in range(1, 29)]

for subj in subjects:
    fpath = os.path.join(CLEAN_DIR, subj, "Cleaned_Fused_MI_MBLL_cnt.mat")
    if not os.path.isfile(fpath):
        print(f"{subj}: fichero no encontrado, omito.")
        continue

    mat  = sio.loadmat(fpath)
    HbO  = mat["HbO"]         # shape (T, nCh)
    HbR  = mat["HbR"]

    if HbO.shape[0] <= padlen:
        print(f"{subj}: señal demasiado corta para filtrar.")
        continue

    # 1) Filtrado
    HbO_f = butter_bandpass(HbO, *BAND, FS, ORDER)
    HbR_f = butter_bandpass(HbR, *BAND, FS, ORDER)

    # 2) Guardar .mat filtrado
    out_dir = os.path.join(OUT_DIR, subj); os.makedirs(out_dir, exist_ok=True)
    sio.savemat(os.path.join(out_dir, "Filtered_Fused_MI_MBLL_cnt.mat"),
                {"HbO": HbO_f, "HbR": HbR_f, "fs": FS,
                 "band":[BAND], "order":ORDER})

    # 3) Diagnóstico rápido RMS
    rms_in  = np.sqrt((HbO**2).mean())
    rms_out = np.sqrt((HbO_f**2).mean())
    print(f"{subj}: filtrado OK  |  RMS {rms_in:.3e} → {rms_out:.3e}")

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


