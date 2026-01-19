# Artifacts_MBLL.py ─ Limpieza de artefactos sobre la señal MI fusionada
import os, numpy as np, scipy.io as sio, matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────── Parámetros ────────────────────────────
DATA_DIR     = Path("Dataset_fNIRS")      # subject xx/cnt.mat
ARTIFACT_DIR = Path("Dataset_fNIRS")      # subject xx/cnt_artifact.mat + mrk_artifact.mat
FUSED_DIR    = Path("MBLL_MI")
OUTPUT_DIR   = Path("Cleaned_Fused_MI")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBJECTS       = [f"subject {i:02}" for i in range(1, 29)]
MI_CELLS       = [0, 2, 4]          # sesiones MI en cnt.mat → motor_imagery 1-3-5
FS             = 10                 # Hz
MARGIN_SAMPLES = 2                  # ±2 muestras extras

# ───────────────────── Función de descarte/interpolación ─────────────────────
def discard_artifacts_in_signal(oxy, deoxy, artifact_positions, artifact_durations,
                                margin=0):
    oxy_cln, deoxy_cln = oxy.copy(), deoxy.copy()
    for onset, dur in zip(artifact_positions, artifact_durations):
        start = max(0, int(onset - margin))
        end   = min(oxy_cln.shape[0], int(onset + dur + margin))
        npt   = end - start
        if npt <= 0:
            continue
        # -- interpolación en cada canal
        for c in range(oxy_cln.shape[1]):
            # HbO
            y  = oxy_cln[:, c]
            y0 = y[start-1] if start > 0 else y[start]
            y1 = y[end]     if end   < len(y) else y[end-1]
            y[start:end] = np.linspace(y0, y1, npt)
            # HbR
            z  = deoxy_cln[:, c]
            z0 = z[start-1] if start > 0 else z[start]
            z1 = z[end]     if end   < len(z) else z[end-1]
            z[start:end] = np.linspace(z0, z1, npt)
    return oxy_cln, deoxy_cln

# ───────────────────────── Bucle por sujeto ─────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

for subj in SUBJECTS:
    print(f"\n── Procesando {subj} ──")

    # 1 · Duraciones de las 6 sesiones originales (cnt.mat) -------------
    cnt_path = os.path.join(DATA_DIR, subj, "cnt.mat")
    if not os.path.isfile(cnt_path):
        print("   cnt.mat no encontrado – omito.")
        continue

    cnt_list = sio.loadmat(cnt_path, squeeze_me=True,
                           struct_as_record=False)["cnt"].flatten()

    # cada elemento de cnt_list es mat_struct con campo .x
    sess_len     = [blk.x.shape[0] for blk in cnt_list]         # [len1..len6]
    full_offset  = np.cumsum([0] + sess_len[:-1])               # inicios absolutos
    mi_len       = [sess_len[i] for i in MI_CELLS]
    mi_offsetAbs = [full_offset[i] for i in MI_CELLS]           # offset en señal completa

    # 2 · Señal MI fusionada (HbO / HbR) -------------------------------
    fused_path = os.path.join(FUSED_DIR, subj, "Fused_MI_MBLL_cnt.mat")
    if not os.path.isfile(fused_path):
        print("   Fused_MI_MBLL_cnt.mat no encontrado – omito.")
        continue

    fdat        = sio.loadmat(fused_path)
    HbO_fused   = fdat["HbO"]            # (T × 24)
    HbR_fused   = fdat["HbR"]
    T_sig       = HbO_fused.shape[0]

    # 3 · Artefactos ----------------------------------------------------
    cntA_path = os.path.join(ARTIFACT_DIR, subj, "cnt_artifact.mat")
    mrkA_path = os.path.join(ARTIFACT_DIR, subj, "mrk_artifact.mat")
    if not (os.path.isfile(cntA_path) and os.path.isfile(mrkA_path)):
        print("   Artefactos no encontrados – omito.")
        continue

    # 3-a) longitud REAL de cada bloque (cnt_artifact)
    cnt_art = sio.loadmat(cntA_path, squeeze_me=True, struct_as_record=False)["cnt_artifact"]
    mrk_art = sio.loadmat(mrkA_path, squeeze_me=True, struct_as_record=False)["mrk_artifact"]

    # --- convierto las celdas MATLAB (1×5) en un array Python de 5 elementos
    deoxy_blocks = np.asarray(cnt_art.deoxy).flatten()   # [blk_EOG, blk_EMG, …]
    mrk_cells    = np.asarray(mrk_art).flatten()         # idem para marcadores

    onset_all, dur_all = [], []
    for blk, m in zip(deoxy_blocks, mrk_cells):          # k = 0..4
        times = np.asarray(m.time).ravel().astype(int)   # onsets
        # Hacemos estimación promedio
        Nt   = blk.x.shape[0] # Muestras totales concatenadas
        est  = max(1, int(round(Nt / max(1,len(times))))) # Duración de cada ocurrencia
        durs = np.full(len(times), est, dtype=int)
        print(f'   {subj}: Se usa estimación promedio ({durs[0]} muestras).')
        onset_all.append(times)
        dur_all.append(durs)

    art_times = np.hstack(onset_all)
    art_durs  = np.hstack(dur_all)

    # 4 · Mapeo a la línea de tiempo MI fusionada -----------------------
    fused_pos, fused_dur = [], []
    for t_full, d_full in zip(art_times, art_durs):
        for idx, L in enumerate(sess_len):
            start_abs = full_offset[idx]
            if start_abs <= t_full < start_abs + L:
                if idx in MI_CELLS:                  # solo MI (1-3-5)
                    # posición relativa dentro de la sesión MI
                    pos_in_block = t_full - start_abs
                    # offset de esa sesión MI dentro de la señal fusionada
                    mi_block     = MI_CELLS.index(idx)
                    pos_fused    = sum(mi_len[:mi_block]) + pos_in_block
                    fused_pos.append(pos_fused)
                    fused_dur.append(d_full)
                break

    fused_pos = np.asarray(fused_pos, dtype=int)

    # (…antes de limpiar, cuando fused_pos y fused_dur ya están calculados…)

    if len(fused_pos):
        # Duraciones que realmente tocan la señal MI
        # Si algún artefacto se “sale” al final de la fusión, se recorta.
        fused_dur = np.minimum(fused_dur,
                            HbO_fused.shape[0] - fused_pos)

        coverage_mi = fused_dur.sum() / HbO_fused.shape[0] * 100
        print(f'Cobertura artefactos en MI: {coverage_mi:.2f}%')
    else:
        print('   (ningún artefacto dentro de las sesiones MI)')


    # --- justo antes del print de diagnóstico --------------------------
    if len(fused_pos):           # sólo si hay artefactos
        print('Artefactos aceptados:', len(fused_pos))
        print('Rango índices artefacto :', fused_pos.min(), '→', fused_pos.max())
        print('Longitud señal fusionada :', HbO_fused.shape[0])
        # Comprobar que ningún onset cae fuera
        assert fused_pos.max() < HbO_fused.shape[0]
    else:
        print('   (ningún artefacto dentro de las sesiones MI)')
        print('Longitud señal fusionada :', HbO_fused.shape[0])


    fused_dur = np.asarray(fused_dur, dtype=int)

    # descarta onsets fuera de rango (por seguridad)
    mask      = fused_pos < T_sig
    fused_pos = fused_pos[mask]
    fused_dur = fused_dur[mask]

    # 5 · Limpieza (interpolación) -------------------------------------
    HbO_cln, HbR_cln = discard_artifacts_in_signal(
        HbO_fused, HbR_fused,
        artifact_positions=fused_pos,
        artifact_durations=fused_dur,
        margin=MARGIN_SAMPLES
    )

    diff = np.abs(HbO_fused - HbO_cln).sum()
    print(f'Diferencia total (norm-1): {diff:.3e}')


    # 6 · Guardar y dibujar -------------------------------------------
    out_dir = os.path.join(OUTPUT_DIR, subj); os.makedirs(out_dir, exist_ok=True)

    # 4.a  comprobación visual  (ponlo justo antes de plt.savefig)
    mask = np.zeros(T_sig, bool)
    for p, d in zip(fused_pos, fused_dur):
        mask[p:p+d] = True
    plt.figure(figsize=(12,2))
    plt.plot(mask.astype(int), lw=1)
    plt.yticks([0,1])
    plt.xlabel('Muestra'); plt.title('Artefactos vs tiempo')
    plt.savefig(os.path.join(out_dir, "Comprobacion.png"))
    plt.close()

    ## 4.b  chequeo de sesiones
    sesiones = np.searchsorted(np.cumsum(mi_len), fused_pos, side='right')
    for k in range(3):
        print(f'Sesión MI {k+1}: {(sesiones==k).sum()} artefactos')


    sio.savemat(os.path.join(out_dir, "Cleaned_Fused_MI_MBLL_cnt.mat"), {
        "HbO": HbO_cln, "HbR": HbR_cln, "fs": FS,
        "artifact_pos": fused_pos, "artifact_dur": fused_dur
    })
    print("   Señal limpia guardada.")

    # gráfico rápido
    plt.figure(figsize=(14,6))
    x = np.arange(T_sig)
    for ch in range(HbO_cln.shape[1]):
        plt.plot(x, HbO_cln[:,ch], label=f"HbO Ch{ch+13}", linestyle='-', color='red',  alpha=.35, lw=.6)
        plt.plot(x, HbR_cln[:,ch], label=f"HbR Ch{ch+13}", linestyle='--', color='blue', alpha=.35, lw=.6)
    for m in np.cumsum(mi_len[:-1]):      # separadores sesiones MI
        plt.axvline(m, color='k', ls=':', lw=.8)
    plt.title(f"{subj} - MI fusionada limpia")
    plt.xlabel("Muestras");  plt.ylabel("mmol/L")
    plt.legend(ncol=2, fontsize="x-small", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Cleaned_Fused_MI_plot.png"))
    plt.close()
    print("   Gráfica guardada.")