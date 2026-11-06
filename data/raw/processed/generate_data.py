import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR)) # lets Python find config.py

import numpy as np
import pandas as pd
import config as cfg


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




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = generate_academics()
    print(df.describe().round(2))
    corr = df.drop(columns="student_id").corr()
    print(corr.round(2))

    cfg.VISUALS_DIR.mkdir(exist_ok=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Correlation of academic features")
    plt.tight_layout()
    plt.savefig(cfg.VISUALS_DIR / "academics_corr_heatmap.png", dpi=150)
    plt.show()

