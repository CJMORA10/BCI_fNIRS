"""
baseline_qc.py  –  Evalúa estabilidad de la línea base (CV y pendiente)
                  en registros fNIRS Nirx (.mat) – canales motores.

* DATA_DIR  : carpeta con “subject 01 … subject 28”.
* MI_CELLS  : índices de las celdas MI   (cnt{1,1}, cnt{1,3}, cnt{1,5} → 0,2,4)
"""

import os
import numpy as np
import scipy.io as sio
from sklearn.linear_model import LinearRegression

# ----------------------------  Configuración  ----------------------------
DATA_DIR   = "Dataset_fNIRS"
SUBJECTS   = [f"subject {i:02}" for i in range(1, 29)]
MI_CELLS   = [0, 2, 4]
FS_EXPECT  = 10          # Hz
MOTOR_LOW  = range(12, 36)
MOTOR_COLS = [i for p in MOTOR_LOW for i in (p, p+36)]   # low,high intercalados

WINDOWS = {                            # nombre -> slice generator
    "10 s": lambda fs: slice(0, 10*fs),
    "30 s": lambda fs: slice(0, 30*fs),
    "60 s": lambda fs: slice(0, 60*fs),
}

# -------------------------  Funciones auxiliares  -------------------------
def unpack(matobj):
    """Desempaqueta 0‑D object arrays de MATLAB."""
    return matobj.item() if isinstance(matobj, np.ndarray) and matobj.ndim == 0 else matobj

def baseline_metrics(data, fs, win_slice):
    """Devuelve (CV %, |pendiente| %/min) promediados entre canales."""
    seg = data[win_slice]
    mean = seg.mean(axis=0)
    std  = seg.std(axis=0)
    cv = (std / mean).mean() * 100                       # %
    # pendiente lineal
    t = (np.arange(seg.shape[0]) / fs).reshape(-1, 1)    # s
    slopes = []
    for ch in range(seg.shape[1]):
        reg = LinearRegression().fit(t, seg[:, ch])
        slopes.append(reg.coef_[0] / mean[ch] * 60 * 100)  # %/min
    return cv, np.mean(np.abs(slopes))

# ------------------------------  Bucle  -----------------------------------
for subj in SUBJECTS:
    cnt_path = os.path.join(DATA_DIR, subj, "cnt.mat")
    if not os.path.isfile(cnt_path):
        continue
    cnt_cells = sio.loadmat(cnt_path, squeeze_me=True)["cnt"]
    print(f"\n### {subj} ###")
    print("Sesión | Ventana |   CV (%) | |pendiente| (%/min)")
    for idx in MI_CELLS:
        cell = cnt_cells[idx]
        fs   = int(cell["fs"].item())
        raw  = unpack(cell["x"])[:, MOTOR_COLS]           # (T, 48)
        for wlabel, slicefn in WINDOWS.items():
            cv, slope = baseline_metrics(raw, fs, slicefn(fs))
            print(f"  {idx+1:>2}   | {wlabel:<6} | {cv:8.3f} | {slope:8.3f}")
