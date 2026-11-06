"""
eligibility.py: rule-based Stage 1 filter.
This is a rule, not an ML model. The same function will power the Streamlit app.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # find config.py

import pandas as pd
import config as cfg


def is_eligible(df):
    """
    Return a True/False Series: does each row meet its company's cutoffs?
    Needs columns: company_category, tenth_pct, twelfth_pct, jee_percentile, cgpa.
    company_category must be 'non_dream' or 'dream' (the keys in cfg.ELIGIBILITY).
    """
    # Two-row table of cutoffs: index = category, columns = tenth, twelfth, jee, cgpa
    cutoffs = pd.DataFrame(cfg.ELIGIBILITY).T

    # For every row, look up the cutoffs of ITS category, aligned to df's index
    cut = cutoffs.loc[df["company_category"]].set_index(df.index)

    return (
        (df["tenth_pct"] > cut["tenth"])
        & (df["twelfth_pct"] > cut["twelfth"])
        & (df["jee_percentile"] > cut["jee"])
        & (df["cgpa"] > cut["cgpa"])
    )