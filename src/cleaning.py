"""cleaning.py: turns placement_raw.csv into a clean table. Returns (clean_df, log)."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import config as cfg

SECTIONS = list(cfg.OA_SECTIONS)
STUDENT_COLS = ["tenth_pct", "twelfth_pct", "jee_percentile", "cgpa"]


def clean_data(raw):
    df = raw.copy()
    log = {"rows_in": len(df)}

    # 1) Exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    log["duplicates_removed"] = log["rows_in"] - len(df)
    assert df["application_id"].is_unique

    # 2) Percentages stored as text ("82%") -> numbers
    for col in ["tenth_pct", "twelfth_pct"]:
        as_text = df[col].astype(str)
        log[f"{col}_text_fixed"] = int(as_text.str.contains("%").sum())
        df[col] = pd.to_numeric(as_text.str.replace("%", "", regex=False).str.strip(),
                                errors="coerce")

    # 3) Impossible percentages (<35 or >100) -> missing, to be recovered in step 5
    for col in ["tenth_pct", "twelfth_pct"]:
        bad = (df[col] < 35) | (df[col] > 100)
        log[f"{col}_impossible"] = int(bad.sum())
        df.loc[bad, col] = np.nan

    # 4) CGPA typos (85 instead of 8.5)
    typo = df["cgpa"] > 10
    log["cgpa_typos_fixed"] = int(typo.sum())
    df.loc[typo, "cgpa"] = df.loc[typo, "cgpa"] / 10
    df.loc[(df["cgpa"] < 0) | (df["cgpa"] > 10), "cgpa"] = np.nan

    # 5) Student-level columns are the same on all of a student's rows,
    #    so fill gaps from the student's other applications
    for col in STUDENT_COLS:
        before = int(df[col].isna().sum())
        df[col] = df[col].fillna(df.groupby("student_id")[col].transform("first"))
        log[f"{col}_recovered_from_other_rows"] = before - int(df[col].isna().sum())
        log[f"{col}_still_missing"] = int(df[col].isna().sum())

    # 6) Flag students whose JEE is truly missing (info the model can use)
    df["jee_missing"] = df["jee_percentile"].isna().astype(int)

    # 7) If exactly ONE OA section is missing, solve it from oa_total (a known weighted sum)
    w = cfg.OA_TOTAL_WEIGHTS
    reached = df["eligible"].astype(bool)
    miss = df[SECTIONS].isna()
    one = reached & (miss.sum(axis=1) == 1)
    for s in SECTIONS:
        rows = one & miss[s]
        others = sum(w[o] * df.loc[rows, o] for o in SECTIONS if o != s)
        df.loc[rows, s] = ((df.loc[rows, "oa_total"] - others) / w[s]).clip(0, 100).round(2)
        log[f"{s}_solved_from_total"] = int(rows.sum())

    # 8) Proper types for the yes/no columns (NaN = stage never reached, kept on purpose)
    for col in ["eligible", "oa_cleared", "selected"]:
        df[col] = df[col].map({True: True, False: False, "True": True, "False": False}) \
                         .astype("boolean")

    return df.sort_values("application_id").reset_index(drop=True), log


if __name__ == "__main__":
    raw = pd.read_csv(cfg.RAW_DIR / "placement_raw.csv")
    clean, log = clean_data(raw)
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(cfg.PROCESSED_DIR / "placement_clean.csv", index=False)
    for k, v in log.items():
        print(f"{k:40s} {v}")
    print("\nSaved clean file:", clean.shape)