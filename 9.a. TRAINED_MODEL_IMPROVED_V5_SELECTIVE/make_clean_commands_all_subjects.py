# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

# Carpeta donde están los CSV de comandos
BASE_DIR = Path("Comandos_V5")
OUT = Path("Comandos_final")
out_dir = OUT
out_dir.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"subject {i:02}" for i in range(1, 29)]


N_SUBJECTS = 28

def clean_one_subject(subj_id: int):
    """
    Lee commands_subjectXX.csv y genera commands_subjectXX_clean.csv
    con columnas extra: cmd_for_unity e is_error.
    """
    sid = f"{subj_id:02d}"
    in_path  = BASE_DIR / SUBJECTS[subj_id - 1] / f"commands_subject{sid}.csv"
    out_path = OUT / f"commands_subject{sid}_clean.csv"

    if not in_path.exists():
        print(f"[AVISO] No se encontró {in_path}, se omite.")
        return

    print(f"=== Procesando subject {sid} ===")
    df = pd.read_csv(in_path)

    # Comprobamos que estén las columnas necesarias
    required_cols = {"true_label", "pred_label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{in_path} no tiene columnas {required_cols}")

    # 0 = idle/rechazo · 1 = right · 2 = left
    def compute_cmd(row):
        true_lab = int(row["true_label"])
        pred_lab = int(row["pred_label"])

        # caso correcto y no rechazo
        if pred_lab != 0 and pred_lab == true_lab:
            return pred_lab
        # rechazos o errores → idle (0)
        return 0

    def compute_is_error(row):
        true_lab = int(row["true_label"])
        pred_lab = int(row["pred_label"])
        # error solo cuando hay decisión (1/2) y es distinta a la verdadera
        return int(pred_lab != 0 and pred_lab != true_lab)

    df["cmd_for_unity"] = df.apply(compute_cmd, axis=1)
    df["is_error"]      = df.apply(compute_is_error, axis=1)

    df.to_csv(out_path, index=False, encoding="utf-8", float_format="%.6f")
    print(f"   → guardado {out_path}")

if __name__ == "__main__":
    for i in range(1, N_SUBJECTS + 1):
        clean_one_subject(i)

    print("\nListo: CSV limpios generados para los sujetos existentes.")
