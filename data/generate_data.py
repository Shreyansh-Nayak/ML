import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR)) # lets Python find config.py

import numpy as np
import pandas as pd
import config as cfg
from src.eligibility import is_eligible


def to_real_scale(z, spec):
    """Convert a standardized z-value into real units and clip to the valid range."""
    values = spec["mean"] + spec["std"] * z
    return np.clip(values, spec["min"], spec["max"])


def generate_academics(n=cfg.N_STUDENTS, seed=cfg.SEED):
    rng = np.random.default_rng(seed)                    # reproducible randomness

    # 10th: the starting signal
    tenth_z = rng.standard_normal(n)

    # 12th depends on 10th
    rho = cfg.ACADEMIC_CORR["twelfth_on_tenth"]
    twelfth_z = rho * tenth_z + np.sqrt(1 - rho**2) * rng.standard_normal(n)

    # JEE depends on 12th
    rho = cfg.ACADEMIC_CORR["jee_on_twelfth"]
    jee_z = rho * twelfth_z + np.sqrt(1 - rho**2) * rng.standard_normal(n)

    # CGPA depends on the combined strength of all three
    combined = (tenth_z + twelfth_z + jee_z) / 3
    combined = (combined - combined.mean()) / combined.std()   # re-standardize
    rho = cfg.ACADEMIC_CORR["cgpa_on_others"]
    cgpa_z = rho * combined + np.sqrt(1 - rho**2) * rng.standard_normal(n)

    df = pd.DataFrame({
        "student_id": np.arange(1, n + 1),
        "tenth_pct": to_real_scale(tenth_z, cfg.ACADEMICS["tenth_pct"]),
        "twelfth_pct": to_real_scale(twelfth_z, cfg.ACADEMICS["twelfth_pct"]),
        "jee_percentile": to_real_scale(jee_z, cfg.ACADEMICS["jee_percentile"]),
        "cgpa": to_real_scale(cgpa_z, cfg.ACADEMICS["cgpa"]),
    })
    return df.round(2)
def build_companies():
    """12 companies: the first N_NON_DREAM are non-dream, the rest dream."""
    n_total = cfg.N_NON_DREAM + cfg.N_DREAM
    categories = ["non_dream"] * cfg.N_NON_DREAM + ["dream"] * cfg.N_DREAM
    return pd.DataFrame({
        "company_id": [f"C{i:02d}" for i in range(1, n_total + 1)],
        "company_category": categories,
    })


def build_applications(academics, seed=cfg.SEED):
    """One row per student-company application, with an 'eligible' flag."""
    rng = np.random.default_rng(seed + 1)   # different stream from academics
    companies = build_companies()
    n_students = len(academics)
    k = cfg.APPLICATIONS_PER_STUDENT

    # Each student picks k DIFFERENT companies:
    # give every (student, company) pair a random number, then take the k smallest per student.
    picks = rng.random((n_students, len(companies))).argsort(axis=1)[:, :k]

    apps = pd.DataFrame({
        "student_id": np.repeat(academics["student_id"].to_numpy(), k),
        "company_id": companies["company_id"].to_numpy()[picks.ravel()],
    })

    # Attach academics and company category. validate= makes pandas raise if keys duplicate.
    apps = apps.merge(academics, on="student_id", how="left", validate="many_to_one")
    apps = apps.merge(companies, on="company_id", how="left", validate="many_to_one")

    apps["eligible"] = is_eligible(apps)
    apps.insert(0, "application_id", np.arange(1, len(apps) + 1))
    return apps


def generate_oa(apps, academics, seed=cfg.SEED):
    """
    Add OA section scores, OA total and oa_cleared to ELIGIBLE applications only.
    Ineligible rows keep NaN, meaning "never reached this stage".
    """
    rng = np.random.default_rng(seed + 2)       # new independent random stream
    w = cfg.OA_SKILL_WEIGHTS

    # 1) One hidden skill value PER STUDENT (not per application)
    def z(s):
        return (s - s.mean()) / s.std()

    raw = (
        w["cgpa"] * z(academics["cgpa"])
        + w["tenth_pct"] * z(academics["tenth_pct"])
        + w["twelfth_pct"] * z(academics["twelfth_pct"])
        + w["jee_percentile"] * z(academics["jee_percentile"])
        + w["random"] * rng.standard_normal(len(academics))   # unmeasured factors
    )
    skill = (raw - raw.mean()) / raw.std()      # re-standardize, same lesson as CGPA
    skill_by_student = pd.Series(skill.to_numpy(), index=academics["student_id"])

    # 2) Copy the student's skill onto each of their applications
    apps = apps.copy()
    mask = apps["eligible"].to_numpy()
    skill_rows = apps.loc[mask, "student_id"].map(skill_by_student).to_numpy()
    n = mask.sum()

    # 3) Section scores: mean + spread * skill + noise (noise is fresh per application)
    for name, p in cfg.OA_SECTIONS.items():
        score = p["mean"] + p["spread"] * skill_rows + p["noise"] * rng.standard_normal(n)
        apps[name] = np.nan
        apps.loc[mask, name] = np.clip(score, 0, 100)

    # 4) OA total = weighted sum of sections
    sections = list(cfg.OA_SECTIONS)
    apps["oa_total"] = np.nan
    apps.loc[mask, "oa_total"] = sum(
        cfg.OA_TOTAL_WEIGHTS[s] * apps.loc[mask, s] for s in sections
    )

    # 5) Cleared = total >= cutoff AND every section >= minimum
    cleared = (apps.loc[mask, "oa_total"] >= cfg.OA_CUTOFF_TOTAL) & (
        apps.loc[mask, sections] >= cfg.OA_MIN_PER_SECTION
    ).all(axis=1)
    apps["oa_cleared"] = pd.Series(pd.NA, index=apps.index, dtype="boolean")
    apps.loc[mask, "oa_cleared"] = cleared.to_numpy()

    return apps.round(2)

def generate_interviews(apps, seed=cfg.SEED):
    """
    Add technical, hr and final_score for applications that cleared the OA.
    Everyone else keeps NaN ("never reached this stage").
    """
    rng = np.random.default_rng(seed + 3)        # new independent random stream
    apps = apps.copy()

    # Rows that cleared the OA (oa_cleared holds True / False / <NA>)
    mask = apps["oa_cleared"].fillna(False).to_numpy(dtype=bool)
    n = mask.sum()
    d = apps.loc[mask]

    # Technical interview: driven by OA total, plus an EXTRA push from coding (plants H2)
    t = cfg.TECHNICAL
    technical = (
        t["base"]
        + t["w_oa"] * d["oa_total"]
        + t["direct_coding_boost_on_technical"] * d["coding"]
        + t["noise"] * rng.standard_normal(n)
    )

    # HR interview: driven by OA total and the "soft" sections
    h = cfg.HR
    hr = (
        h["base"]
        + h["w_oa"] * d["oa_total"]
        + h["w_soft"] * (d["english"] + d["behavioral"]) / 2
        + h["noise"] * rng.standard_normal(n)
    )

    apps["technical"] = np.nan
    apps["hr"] = np.nan
    apps.loc[mask, "technical"] = np.clip(technical, 0, 100).to_numpy()
    apps.loc[mask, "hr"] = np.clip(hr, 0, 100).to_numpy()

    # Final score: weighted sum, rounded to an integer so ties can happen (H3)
    w = cfg.FINAL_SCORE_WEIGHTS
    final = (
        w["oa"] * apps.loc[mask, "oa_total"]
        + w["technical"] * apps.loc[mask, "technical"]
        + w["hr"] * apps.loc[mask, "hr"]
    )
    apps["final_score"] = np.nan
    if cfg.ROUND_FINAL_SCORE_TO_INT:
        final = final.round(0)
    apps.loc[mask, "final_score"] = final.to_numpy()

    apps[["technical", "hr"]] = apps[["technical", "hr"]].round(2)
    return apps

def h2_check(apps):
    """Does coding predict technical BEYOND oa_total? (coefficient should be near the planted 0.15)"""
    from sklearn.linear_model import LinearRegression
    d = apps[apps["technical"].notna()]
    X = d[["oa_total", "coding"]]
    m = LinearRegression().fit(X, d["technical"])
    return dict(zip(X.columns, m.coef_.round(3)))
def select_candidates(apps):
    """
    Pick the top VACANCIES_PER_COMPANY candidates per company by final_score.
    Ties are broken by coding (higher wins), then application_id (so results are reproducible).
    Adds a 'selected' column: True / False for interviewed rows, <NA> for rows that never reached interviews.
    Returns (apps, h3_info) where h3_info describes how often the tie-break decided the last seat.
    """
    apps = apps.copy()
    v = cfg.VACANCIES_PER_COMPANY

    iv = apps[apps["final_score"].notna()].copy()
    iv = iv.sort_values(
        ["company_id", "final_score", cfg.TIEBREAK_COLUMN, "application_id"],
        ascending=[True, False, False, True],
    )
    iv["rank"] = iv.groupby("company_id").cumcount()      # 0 = best in that company
    iv["selected"] = iv["rank"] < v

    # ---- H3: did the tie-break decide the last seat? ----
    # Compare the last selected candidate (rank v-1) with the first rejected one (rank v).
    last_in = iv[iv["rank"] == v - 1].set_index("company_id")
    first_out = iv[iv["rank"] == v].set_index("company_id")
    common = last_in.index.intersection(first_out.index)   # companies that had more than v candidates

    tied = last_in.loc[common, "final_score"] == first_out.loc[common, "final_score"]
    # The tie-break only mattered if the coding scores differed (otherwise application_id decided)
    coding_decided = tied & (
        last_in.loc[common, cfg.TIEBREAK_COLUMN] != first_out.loc[common, cfg.TIEBREAK_COLUMN]
    )
    h3_info = {
        "companies_with_competition": len(common),
        "last_seat_tied_on_final_score": int(tied.sum()),
        "decided_by_coding_tiebreak": int(coding_decided.sum()),
        "decided_by_application_id": int((tied & ~coding_decided).sum()),
    }

    apps["selected"] = pd.Series(pd.NA, index=apps.index, dtype="boolean")
    apps.loc[iv.index, "selected"] = iv["selected"]
    return apps, h3_info

def plot_academics_heatmap(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    corr = df.drop(columns="student_id").corr()
    cfg.VISUALS_DIR.mkdir(exist_ok=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Correlation of academic features")
    plt.tight_layout()
    plt.savefig(cfg.VISUALS_DIR / "academics_corr_heatmap.png", dpi=150)
    plt.close()


def r2_oa_vs_academics(apps):
    """How much of OA total do the four academic columns explain? (eligible rows)"""
    from sklearn.linear_model import LinearRegression
    d = apps[apps["eligible"]]
    X = d[["tenth_pct", "twelfth_pct", "jee_percentile", "cgpa"]]
    return LinearRegression().fit(X, d["oa_total"]).score(X, d["oa_total"])


if __name__ == "__main__":
    # 1) Create the data, one stage at a time (each line uses the previous result)
    academics = generate_academics()
    plot_academics_heatmap(academics)
    apps = build_applications(academics)      # <- 'apps' is born here
    apps = generate_oa(apps, academics)       # <- and updated here
    apps = generate_interviews(apps)          # <- only now can this line run
    apps, h3 = select_candidates(apps)
    # 2) Funnel checks
    print("Funnel:")
    print("  applications :", len(apps))
    print("  eligible     :", int(apps["eligible"].sum()))
    print("  OA cleared   :", int(apps["oa_cleared"].sum()))
    print("\nOA cleared by category:")
    print(apps[apps["eligible"]].groupby("company_category")["oa_cleared"]
          .agg(cleared="sum", eligible="count"))
    print("\nOA cleared per dream company:")
    print(apps[apps.company_category == "dream"].groupby("company_id")["oa_cleared"].sum())
    print("\nMean section scores (eligible):")
    print(apps.loc[apps.eligible, list(cfg.OA_SECTIONS) + ["oa_total"]].mean().round(1))
    print("\nR2 of OA total vs academics:", round(r2_oa_vs_academics(apps), 3))
    
    # 3) Interview stage checks
    iv = apps[apps["technical"].notna()]
    print("\nInterview stage rows:", len(iv))
    print(iv[["oa_total", "technical", "hr", "final_score"]].describe().round(1))
    print("\nDistinct final scores:", iv["final_score"].nunique())
    print("H2 coefficients (technical ~ oa_total + coding):", h2_check(apps))
    print("technical clipped at 100:", (iv["technical"] >= 100).sum())


        # 4) Selection checks
    print("\nSelected total:", int(apps["selected"].sum()), "(expected", cfg.TARGETS["selected_total_expected"], ")")
    print("Selected per company:")
    print(apps[apps["selected"].notna()].groupby("company_id")["selected"].sum())
    print("\nH3 tie-break summary:", h3)

        # 5) Save the clean "answer key" (before any mess is added)
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.RAW_DIR / "placement_truth.csv"
    apps.to_csv(out_path, index=False)
    print("\nSaved:", out_path, "| rows:", len(apps), "| columns:", len(apps.columns))