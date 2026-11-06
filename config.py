"""
config.py: single source of truth for the simulated placement process.
Change a number here, regenerate the data, and everything downstream updates.
"""
from pathlib import Path

# ---------- Reproducibility ----------
SEED = 42
N_STUDENTS = 3000

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
VISUALS_DIR = ROOT / "visuals"

# ---------- Stage 0: academic profile ----------
# Each value is generated from the previous one plus noise, which creates the correlations.
ACADEMICS = {
    "tenth_pct":      {"mean": 80, "std": 8, "min": 45, "max": 99},
    "twelfth_pct":    {"mean": 78, "std": 9, "min": 45, "max": 99},
    "jee_percentile": {"mean": 80, "std": 12, "min": 30, "max": 99.9},
    "cgpa":           {"mean": 7.6, "std": 0.9, "min": 5.0, "max": 10.0},
}
# Target correlations (check the heatmap against these: aim for 0.3-0.6)
ACADEMIC_CORR = {
    "twelfth_on_tenth": 0.6,
    "jee_on_twelfth": 0.5,
    "cgpa_on_others": 0.4,
}

# ---------- Companies ----------
N_NON_DREAM = 8
N_DREAM = 4
APPLICATIONS_PER_STUDENT = 5        # average, among companies the student is eligible for
VACANCIES_PER_COMPANY = 8

# ---------- Stage 1: eligibility rules (rule-based filter) ----------
ELIGIBILITY = {
    "non_dream": {"tenth": 70, "twelfth": 70, "jee": 80, "cgpa": 7.5},
    "dream":     {"tenth": 80, "twelfth": 80, "jee": 90, "cgpa": 8.5},
}

# ---------- Stage 2: Online Assessment ----------
# How strongly academics drive the hidden "skill" signal (must sum to 1.0)
OA_SKILL_WEIGHTS = {
    "cgpa": 0.50,
    "tenth_pct": 0.15,
    "twelfth_pct": 0.15,
    "jee_percentile": 0.05,
    "random": 0.15,
}
# Section score = mean + spread * skill_signal + noise
OA_SECTIONS = {
    #              mean  spread  noise_std
    "logical":    {"mean": 62, "spread": 12, "noise": 10},
    "quant":      {"mean": 60, "spread": 13, "noise": 10},
    "coding":     {"mean": 58, "spread": 15, "noise": 9},
    "english":    {"mean": 65, "spread": 8,  "noise": 13},
    "behavioral": {"mean": 66, "spread": 6,  "noise": 14},
}
# Weights for the OA total (must sum to 1.0)
OA_TOTAL_WEIGHTS = {"logical": 0.20, "quant": 0.20, "coding": 0.30,
                    "english": 0.15, "behavioral": 0.15}
OA_CUTOFF_TOTAL = 65        # % of normalized OA total (0-100)
OA_MIN_PER_SECTION = 40     # minimum % required in every section

# ---------- Stage 3: interviews ----------
# technical = base + w_oa*OA_total + coding_boost*coding + noise
TECHNICAL = {"base": 10, "w_oa": 0.65, "direct_coding_boost_on_technical": 0.15, "noise": 10}
# hr = base + w_oa*OA_total + w_soft*(english+behavioral)/2 + noise
HR = {"base": 15, "w_oa": 0.30, "w_soft": 0.35, "noise": 12}

FINAL_SCORE_WEIGHTS = {"oa": 0.40, "technical": 0.35, "hr": 0.25}
ROUND_FINAL_SCORE_TO_INT = True     # makes ties possible (hypothesis 3)
TIEBREAK_COLUMN = "coding"          # higher coding score wins a tie

# ---------- Deliberate mess (so cleaning has something to do) ----------
MESS = {
    "jee_missing_rate": 0.08,        # students with no JEE score
    "cgpa_typo_rate": 0.005,         # e.g. 85 instead of 8.5
    "pct_as_string_rate": 0.03,      # "82%" instead of 82
    "duplicate_rows_rate": 0.01,
    "random_missing_rate": 0.02,     # random NaN in other columns
    "outlier_rate": 0.005,
}

# ---------- Sanity checks after generation ----------
TARGETS = {
    "oa_r2_vs_academics": (0.50, 0.65),
    "selected_total_expected": (N_NON_DREAM + N_DREAM) * VACANCIES_PER_COMPANY,
}

# ---------- Guard rails: fail loudly if weights are wrong ----------
assert abs(sum(OA_SKILL_WEIGHTS.values()) - 1) < 1e-9, "OA skill weights must sum to 1"
assert abs(sum(OA_TOTAL_WEIGHTS.values()) - 1) < 1e-9, "OA total weights must sum to 1"
assert abs(sum(FINAL_SCORE_WEIGHTS.values()) - 1) < 1e-9, "Final weights must sum to 1"