"""
predict.py: prediction system for one student.
Stage 1 (academics only): eligibility + expected OA score.
Stage 2 (academics + OA sections): OA cleared?, selection score, expected final score.
Decision-support tool trained on SYNTHETIC data. It is not a hiring judge.
"""
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import config as cfg
from src.eligibility import is_eligible

SECTIONS = list(cfg.OA_SECTIONS)
_cache = {}


def _models():
    """Load the saved models once, then reuse them."""
    if not _cache:
        d = cfg.MODELS_DIR
        _cache["stage1"] = joblib.load(d / "stage1_oa_regressor.joblib")
        _cache["clf"] = joblib.load(d / "stage2_selection_classifier.joblib")
        _cache["final"] = joblib.load(d / "stage2_final_score_regressor.joblib")
        _cache["features"] = json.load(open(d / "feature_lists.json"))
    return _cache


def _check_range(name, value, lo, hi):
    if not (lo <= value <= hi):
        raise ValueError(f"{name}={value} is outside the valid range {lo}-{hi}")


def _academic_row(student, category):
    """One-row DataFrame in the format the models and is_eligible expect."""
    jee = student.get("jee_percentile")            # optional
    _check_range("tenth_pct", student["tenth_pct"], 0, 100)
    _check_range("twelfth_pct", student["twelfth_pct"], 0, 100)
    _check_range("cgpa", student["cgpa"], 0, 10)
    if jee is not None:
        _check_range("jee_percentile", jee, 0, 100)
    if category not in cfg.ELIGIBILITY:
        raise ValueError(f"category must be one of {list(cfg.ELIGIBILITY)}")
    return pd.DataFrame([{
        "tenth_pct": student["tenth_pct"],
        "twelfth_pct": student["twelfth_pct"],
        "jee_percentile": np.nan if jee is None else jee,
        "cgpa": student["cgpa"],
        "jee_missing": int(jee is None),
        "company_category": category,
    }])


def predict_stage1(student, category="non_dream"):
    """Eligibility check (rules) + expected OA total (regression)."""
    row = _academic_row(student, category)
    eligible = bool(is_eligible(row).iloc[0])
    out = {"category": category, "eligible": eligible, "expected_oa_total": None}

    if row["jee_missing"].iloc[0] == 1:
        out["note"] = "JEE score missing: eligibility cannot be confirmed (JEE is a required cutoff)."
    if not eligible:
        out.setdefault("note", "Does not meet this category's academic cutoffs.")
        out["gaps"] = {k: v for k, v in _gaps(row.iloc[0], category).items()}
        return out

    m = _models()
    pred = float(m["stage1"].predict(row[m["features"]["stage1"]])[0])
    out["expected_oa_total"] = round(float(np.clip(pred, 0, 100)), 1)
    out["oa_cutoff"] = cfg.OA_CUTOFF_TOTAL
    out["note"] = "Typical error of this estimate is about 4-5 points (cross-validated MAE 4.4)."
    return out


def _gaps(r, category):
    """How far each academic value is below the cutoff (only the ones that fail)."""
    cut = cfg.ELIGIBILITY[category]
    pairs = {"tenth_pct": "tenth", "twelfth_pct": "twelfth",
             "jee_percentile": "jee", "cgpa": "cgpa"}
    gaps = {}
    for col, key in pairs.items():
        v = r[col]
        if pd.isna(v):
            gaps[col] = "missing"
        elif v <= cut[key]:
            gaps[col] = round(float(cut[key] - v), 2)
    return gaps


def predict_stage2(student, oa_scores, category="non_dream"):
    """After the OA: did it clear? Then selection score and expected final score."""
    for s in SECTIONS:
        if s not in oa_scores:
            raise ValueError(f"missing OA section: {s}")
        _check_range(s, oa_scores[s], 0, 100)

    oa_total = sum(cfg.OA_TOTAL_WEIGHTS[s] * oa_scores[s] for s in SECTIONS)
    low = [s for s in SECTIONS if oa_scores[s] < cfg.OA_MIN_PER_SECTION]
    cleared = oa_total >= cfg.OA_CUTOFF_TOTAL and not low
    out = {"oa_total": round(oa_total, 1), "oa_cleared": bool(cleared)}

    if not cleared:
        out["note"] = (f"OA total below {cfg.OA_CUTOFF_TOTAL}" if oa_total < cfg.OA_CUTOFF_TOTAL else "")
        if low:
            out["note"] += f" Sections below the {cfg.OA_MIN_PER_SECTION} minimum: {low}"
        out["note"] = out["note"].strip()
        return out

    row = _academic_row(student, category)
    for s in SECTIONS:
        row[s] = oa_scores[s]
    m = _models()
    X = row[m["features"]["stage2"]]
    out["selection_score"] = round(float(m["clf"].predict_proba(X)[0, 1]), 3)
    out["expected_final_score"] = round(float(np.clip(m["final"].predict(X)[0], 0, 100)), 1)
    out["note"] = ("selection_score is a ranking score, not a true probability: the model was trained "
                   "with class_weight='balanced', which inflates it. Interviews are not observed yet.")
    return out


if __name__ == "__main__":
    student = {"tenth_pct": 88, "twelfth_pct": 85, "jee_percentile": 93, "cgpa": 8.9}
    oa = {"logical": 72, "quant": 70, "coding": 78, "english": 66, "behavioral": 68}

    print("Stage 1 (non-dream):", predict_stage1(student, "non_dream"))
    print("Stage 1 (dream)    :", predict_stage1(student, "dream"))
    print("Stage 2            :", predict_stage2(student, oa, "dream"))
    print("No JEE             :", predict_stage1({**student, "jee_percentile": None}))
    print("Weak student       :", predict_stage1({"tenth_pct": 65, "twelfth_pct": 68,
                                                  "jee_percentile": 70, "cgpa": 7.0}))