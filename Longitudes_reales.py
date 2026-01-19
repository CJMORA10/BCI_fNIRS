# file: check_wavelengths.py
# -----------------------------------------
import os, csv
import numpy as np
import scipy.io as sio

DATA_DIR = "Dataset_fNIRS"                       # raíz de los sujetos
SUBJECTS = [f"subject {i:02}" for i in range(1, 29)]
OUT_CSV  = "wavelengths_por_sesion.csv"

rows = []  # (subject, session, title, λ_low, λ_high, fs_Hz, n_channels)

def unpack(arr):
    """Desempaca 0‑D object arrays de MATLAB a np.ndarray normal."""
    if isinstance(arr, np.ndarray) and arr.ndim == 0:
        arr = arr.item()
    return np.asarray(arr)

for subj in SUBJECTS:
    cnt_path = os.path.join(DATA_DIR, subj, "cnt.mat")
    if not os.path.isfile(cnt_path):
        print(f"[ADVERTENCIA] {subj}: cnt.mat no encontrado")
        continue

    cnt_cells = sio.loadmat(cnt_path, squeeze_me=True)["cnt"]

    for idx, cell in enumerate(cnt_cells, start=1):
        # --- título (MI, MA, artefact) ---
        title = str(cell["title"].item()) if "title" in cell.dtype.names else "?"

        # --- longitudes de onda ---
        wl_raw = unpack(cell["wavelengths"])
        if wl_raw.size != 2:
            print(f"[ALERTA] {subj} sesión {idx}: wavelengths raro → {wl_raw}")
            continue
        λ_low, λ_high = map(int, wl_raw)

        # --- FS y canales (control) ---
        fs = int(cell["fs"].item())

        clab_raw = unpack(cell["clab"])          # desempaquetar cell array
        n_ch = clab_raw.size if isinstance(clab_raw, np.ndarray) else len(clab_raw)

        rows.append([subj, idx, title, λ_low, λ_high, fs, n_ch])

# ---------------- Imprimir resumen ----------------
print("\nResumen λ por sesión")
print("Subject  Sesión  Tipo  λ_low  λ_high  fs  canales")
for r in rows:
    print(f"{r[0]:>8}   {r[1]:>2}     {r[2]:<3}   {r[3]:>4}   {r[4]:>4}   {r[5]:>2}   {r[6]:>3}")

# ---------------- Guardar CSV ----------------
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Subject", "Session", "Title", "Lambda_low_nm",
                "Lambda_high_nm", "fs_Hz", "n_channels"])
    w.writerows(rows)

print(f"\nCSV con todas las sesiones guardado en: {OUT_CSV}")
