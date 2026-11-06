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


if __name__ == "__main__":
    academics = generate_academics()
    plot_academics_heatmap(academics)

    apps = build_applications(academics)
    print("Total applications:", len(apps))
    print("\nEligible applications by category:")
    print(apps.groupby("company_category")["eligible"].sum())
    print("\nEligible applications per company:")
    print(apps.groupby("company_id")["eligible"].sum())
    print("\nApplications per student (should all be", cfg.APPLICATIONS_PER_STUDENT, "):")
    print(apps.groupby("student_id").size().value_counts())
