"""Streamlit front end for the placement prediction system (wraps src/predict.py)."""
import sys
from math import erf, sqrt
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
import config as cfg
from src.predict import predict_stage1, predict_stage2, SECTIONS

OA_RMSE = 5.45          # cross-validated RMSE of the Stage 1 model (from your evaluation)
BASE_RATE = 0.10        # share of OA-clearers who were selected (96 / 960)

st.set_page_config(page_title="Placement Readiness Predictor", page_icon="🎯", layout="wide")
st.title("🎯 Placement Readiness Predictor")
st.caption("Decision-support prototype trained on SYNTHETIC data. It shows readiness gaps; "
           "it is not a hiring judge and says nothing about real companies.")


def clear_chance(expected_oa):
    """Rough chance that the true OA total is >= cutoff, assuming errors are roughly normal."""
    z = (cfg.OA_CUTOFF_TOTAL - expected_oa) / OA_RMSE
    return 1 - 0.5 * (1 + erf(z / sqrt(2)))

def readiness_verdict(r1):
    """Plain-language summary for Stage 1. Wording is 'readiness', never a guarantee."""
    if not r1["eligible"]:
        return "error", "Not eligible for this company type yet. See the gaps below."
    p = clear_chance(r1["expected_oa_total"])
    if p >= 0.70:
        return "success", f"Strong readiness: likely to clear the OA (about {p:.0%} under this simulation)."
    if p >= 0.40:
        return "warning", f"Moderate readiness: the OA is a coin flip (about {p:.0%}). Coding and quant are the heaviest sections."
    return "error", f"Low readiness: clearing the OA is unlikely (about {p:.0%}). Eligible, but the OA is a real hurdle."


# ---------------- Sidebar: the student's academic profile ----------------
with st.sidebar:
    st.header("Academic profile")
    tenth = st.number_input("10th %", 0.0, 100.0, 88.0, 0.5)
    twelfth = st.number_input("12th %", 0.0, 100.0, 85.0, 0.5)
    cgpa = st.number_input("CGPA (out of 10)", 0.0, 10.0, 8.9, 0.1)
    has_jee = st.checkbox("I have a JEE percentile", value=True)
    jee = st.number_input("JEE percentile", 0.0, 100.0, 93.0, 0.5) if has_jee else None
    category = st.radio("Company type", list(cfg.ELIGIBILITY),
                        format_func=lambda c: c.replace("_", "-").title())

student = {"tenth_pct": tenth, "twelfth_pct": twelfth, "jee_percentile": jee, "cgpa": cgpa}
tab1, tab2, tab3,tab4 = st.tabs(["1. Before the OA", "2. After the OA", "3. What-if simulator", "4. Model Performance"])

# ---------------- Tab 1: eligibility + expected OA score ----------------
with tab1:
    try:
        r1 = predict_stage1(student, category)
    except ValueError as err:
        st.error(str(err))
        r1 = None

    if r1:
        level, msg = readiness_verdict(r1)
        getattr(st, level)(f"**Verdict:** {msg}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Eligible?", "Yes ✅" if r1["eligible"] else "No ❌")
        if r1["eligible"]:
            exp = r1["expected_oa_total"]
            c2.metric("Expected OA total", f"{exp}", f"{exp - cfg.OA_CUTOFF_TOTAL:+.1f} vs cutoff {cfg.OA_CUTOFF_TOTAL}")
            c3.metric("Rough chance of clearing OA", f"{clear_chance(exp):.0%}")
            st.progress(min(max(exp / 100, 0.0), 1.0), text=f"Expected {exp} / 100 (cutoff {cfg.OA_CUTOFF_TOTAL})")
            st.info(r1["note"] + " The clearing chance is a rough estimate that assumes normally distributed errors.")
        else:
            st.warning(r1["note"])
            if r1.get("gaps"):
                st.write("**Gap to the cutoffs** (points below the requirement):")
                st.table(pd.DataFrame(r1["gaps"].items(), columns=["feature", "gap"]).set_index("feature"))

# ---------------- Tab 2: after the OA ----------------
with tab2:
    st.write("Enter the OA section scores (0-100).")
    cols = st.columns(len(SECTIONS))
    oa = {s: cols[i].slider(s.title(), 0, 100, 65) for i, s in enumerate(SECTIONS)}
    try:
        r2 = predict_stage2(student, oa, category)
    except ValueError as err:
        st.error(str(err))
        r2 = None

    if r2:
        c1, c2 = st.columns(2)
        c1.metric("OA total", r2["oa_total"])
        c2.metric("OA cleared?", "Yes ✅" if r2["oa_cleared"] else "No ❌")

        if r2["oa_cleared"]:
            score = r2["selection_score"]
            ratio = score / BASE_RATE

            c3, c4 = st.columns(2)
            c3.metric("Selection score", score, f"{ratio:.1f}x the typical OA-clearer")
            c4.metric("Expected final score", r2["expected_final_score"])
            st.info(r2["note"])

            if ratio >= 2:
                st.success(f"Verdict: competitive. Selection score is {ratio:.1f}x the typical OA-clearer.")
            else:
                st.warning(f"Verdict: average or below ({ratio:.1f}x typical). Interviews will decide.")
        else:
            st.warning(r2["note"])

# ---------------- Tab 3: what-if simulator ----------------
with tab3:
    st.write("See how the expected OA total changes as one academic value changes. "
             "Everything else stays as in the sidebar.")
    var = st.selectbox("Vary", ["cgpa", "tenth_pct", "twelfth_pct", "jee_percentile"])
    lo, hi = (5.0, 10.0) if var == "cgpa" else (50.0, 99.0)
    xs = [round(lo + i * (hi - lo) / 40, 2) for i in range(41)]
    ys = []
    for x in xs:
        s = {**student, var: x}
        if s["jee_percentile"] is None:
            ys.append(None); continue
        try:
            ys.append(predict_stage1(s, category)["expected_oa_total"])
        except ValueError:
            ys.append(None)
    sweep = pd.DataFrame({var: xs, "expected_oa_total": ys, "oa_cutoff": cfg.OA_CUTOFF_TOTAL}).set_index(var)
    st.line_chart(sweep)
    st.caption("The line only starts where the student meets the company's academic cutoffs (rule-based eligibility). "
               "Below that point the model is not used.")

with tab4:
    st.subheader("Does the model actually predict?")
    bt = pd.read_csv(cfg.PROCESSED_DIR / "backtest_oa.csv")

    pred_clear = bt["pred_oa"] >= cfg.OA_CUTOFF_TOTAL
    true_clear = bt["oa_total"] >= cfg.OA_CUTOFF_TOTAL
    tp = int((pred_clear & true_clear).sum())
    c1, c2 = st.columns(2)
    c1.metric("Precision at cutoff", f"{tp / pred_clear.sum():.2f}",
              help="Of applications predicted to clear the OA, the share that did")
    c2.metric("Recall at cutoff", f"{tp / true_clear.sum():.2f}",
              help="Of applications that cleared, the share the model caught")

    # Predicted vs actual with a y = x reference line
    plot = bt.sample(600, random_state=1)[["pred_oa", "oa_total"]].copy()
    plot["perfect_prediction"] = plot["pred_oa"]
    st.scatter_chart(plot, x="pred_oa", y=["oa_total", "perfect_prediction"])
    st.caption("Each point is an application the model never saw (out-of-sample, grouped by student). "
               "Points near the 'perfect_prediction' line are accurate. One student can appear several times.")

    st.markdown("**Regression results (5-fold grouped CV)**")
    st.dataframe(pd.read_csv(cfg.PROCESSED_DIR / "regression_results.csv"), hide_index=True)
    st.markdown("**Selection models A-D (AUC)**")
    st.dataframe(pd.read_csv(cfg.PROCESSED_DIR / "model_comparison.csv"), hide_index=True)
    st.image(str(cfg.VISUALS_DIR / "07_model_comparison_auc.png"), use_container_width=True)

    st.warning("Trained on SYNTHETIC data. These numbers show the method works on a process I "
               "simulated. They do not show how real placements behave.")

