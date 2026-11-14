"""features.py: single definition of which columns each model may use."""
import config as cfg

ACAD = ["tenth_pct", "twelfth_pct", "jee_percentile", "cgpa", "jee_missing"]
OA = list(cfg.OA_SECTIONS)
INTERVIEW = ["technical", "hr"]

FEATURE_SETS = {
    "A": ACAD,
    "B": ACAD + OA,
    "C": ACAD + INTERVIEW,
    "D": ACAD + OA + INTERVIEW,
}

# Never usable as features: they are outcomes, sums of features, labels or identifiers
EXCLUDED = {
    "final_score": "outcome used to pick the selected candidates (direct leakage)",
    "oa_total": "weighted sum of the five OA sections (redundant)",
    "oa_cleared": "outcome of the OA stage",
    "eligible": "outcome of rule-based stage 1",
    "selected": "the target",
    "application_id": "identifier",
    "student_id": "identifier (used only for grouped splits)",
    "company_id": "label of the company, not a student attribute",
    "company_category": "label of the company, not a student attribute",
}