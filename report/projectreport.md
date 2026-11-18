# Placement Readiness Prediction System: Project Report

**Author:** [Shreyansh Nayak]  |  
**Repository:** [https://github.com/Shreyansh-Nayak/ML]

---

## Abstract

This project builds a prototype machine-learning system that models a campus placement funnel for engineering students (eligibility → online assessment → interviews → final selection) and predicts outcomes from the information available at each stage. Because real placement data is private, a simulator generated 3,000 students and 15,000 student-company applications with deliberate data-quality problems. The data was cleaned, explored, and used to train regression and classification models under grouped cross-validation. Academics alone predict selection with AUC 0.79, adding OA scores raises it to 0.90, and adding interview scores to 0.97-0.99. OA total is predicted from academics with R² 0.64 (MAE 4.4 vs 7.3 for a baseline). Three hypotheses were tested: H1 was rejected, H2 supported, and H3 supported as a frequency. A Streamlit app wraps the models as a readiness tool. Since the data is synthetic, the results validate the method, not real-world placement behaviour.

---

## 1. Introduction and Objective

During placements, students face a multi-stage process they cannot see into: cutoffs, an online assessment with several sections, and interview rounds. They rarely know whether they are eligible, how ready they are, or what to improve. Placement cells likewise lack a stage-wise view of where candidates drop out.

**Objective:** build and evaluate a beginner-level ML system that analyses student data and predicts student performance in the placement context, covering the full workflow: Data → Cleaning → Analysis → Feature Selection → Training → Testing → Evaluation → Prediction.

**Framing:** "student performance" is interpreted as placement performance. The system is decision support for students and placement cells, not an automated hiring judge.

## 2. Problem Formulation

| Stage | Known information | Regression target | Classification target |
|---|---|---|---|
| 1. Readiness | 10th %, 12th %, JEE, CGPA | OA total | Will clear the OA cutoff? |
| 2. Post-OA | + 5 OA sections | Final score | Selected or not (Model B) |
| 3. Post-interview | + technical, HR | Final score | Selected or not (Models C, D) |

Eligibility is a **rule-based filter**, not an ML task. Treating the company's rules as the simulated process and ML as prediction from partial information avoids a trivial result (a model rediscovering an if-statement).

## 3. Dataset

### 3.1 Why synthetic

Real placement data is private and a survey would have delayed the project. The assignment allows creating a dataset. Synthetic data cannot show real-world insights, but it can show the pipeline works end to end, and because the true weights are known, it allows testing whether the models recover them.

### 3.2 Generation process (`data/generate_data.py`, seed 42, parameters in `config.py`)

1. **Academics.** 3,000 students. 10th %, then 12th % = f(10th) + noise, then JEE percentile = f(12th) + noise, then CGPA = f(combined) + noise, using `z = rho·prev + sqrt(1−rho²)·noise` on standardised values. Achieved correlations: 10th-12th 0.60, 12th-JEE 0.49, CGPA with others about 0.31-0.34.
2. **Applications.** 12 companies (8 non-dream, 4 dream). Each student applies to 5 different companies, giving 15,000 rows. Eligibility is then checked.
3. **Online assessment.** One hidden skill value per student: 0.50·CGPA + 0.15·10th + 0.15·12th + 0.05·JEE + 0.15·randomness (all standardised, then re-standardised). Section score = mean + spread·skill + per-application noise, clipped to 0-100. Clear if total ≥ 65 and every section ≥ 40.
4. **Interviews.** Technical = 4 + 0.65·OA + 0.25·coding + noise(10). HR = 15 + 0.30·OA + 0.35·(English+Behavioral)/2 + noise(12).
5. **Selection.** Final = 0.40·OA + 0.35·Technical + 0.25·HR, rounded to an integer. Top 8 per company; ties broken by coding score, then application ID.
6. **Mess injection.** See Section 5.

### 3.3 Calibration decisions

| Observation | Decision |
|---|---|
| Only 37 of 1,000 students were dream-eligible (32 seats) | Raised students to 3,000 and applications to 5, instead of loosening company rules |
| First OA run cleared 80% (99% for dream): eligible students have skill about +1 SD, section means were set for an average student | Lowered all section means by 12 points. Result: 29.8% non-dream, 70% dream; R² 0.643 |
| Coding effect on technical came out at 0.058 (planted 0.15) | Raised planted boost to 0.25 and lowered technical base to 4. Estimate then 0.157. Stopped tuning to avoid fitting the answer |

### 3.4 Funnel

| Stage | Count |
|---|---|
| Applications | 15,000 |
| Eligible | 2,990 (2,819 non-dream, 171 dream) |
| Cleared OA | 960 (840 non-dream, 120 dream) |
| Selected | 96 (10% of interviewed) |

### 3.5 Data dictionary (clean file: 15,000 rows × 21 columns)

| Column | Type | Description | Stage |
|---|---|---|---|
| application_id | int | Unique student-company application | ID |
| student_id | int | Student (join key, up to 5 rows each) | ID |
| company_id, company_category | str | Company and `dream` / `non_dream` | ID |
| tenth_pct, twelfth_pct | float | Board percentages | 1 |
| jee_percentile | float | JEE percentile (missing for 253 students) | 1 |
| cgpa | float | Aggregate CGPA (0-10) | 1 |
| eligible | bool | Meets company cutoffs (rule) | 1 |
| logical, quant, coding, english, behavioral | float | OA section scores (0-100) | 2 |
| oa_total | float | Weighted OA total | 2 |
| oa_cleared | bool | OA total ≥ 65 and all sections ≥ 40 | 2 |
| technical, hr | float | Interview scores (0-100) | 3 |
| final_score | float | Weighted final score (integer) | 3 |
| selected | bool | Top 8 at the company | 3 |
| jee_missing | int | 1 if the student has no JEE value (added in cleaning) | derived |

NaN in stage columns means the application never reached that stage. This is intentional, not an error.

## 4. Methodology Overview

Reproducibility: fixed seeds, all parameters in `config.py`, one data flow (`raw → processed → models`), raw data never edited by hand. The clean file `placement_truth.csv` is kept as an answer key to validate cleaning; models never train on it.

## 5. Data Cleaning

### 5.1 Injected mess and cleaning results

| Problem | Injected | Cleaning step | Result |
|---|---|---|---|
| Exact duplicate rows | 146 | `drop_duplicates` first | 15,146 → 15,000 rows |
| Percentages as text ("82%") | 858 cells | strip `%`, convert to number | 408 + 450 fixed |
| Impossible percentages (<0, >100) | 86 | set to missing | 43 + 42 blanked |
| CGPA typo (85 for 8.5) | 79 | divide by 10 when > 10 | 77 fixed |
| Missing 10th, 12th, CGPA | random gaps and blanked outliers | fill from the same student's other rows | 364, 321, 288 recovered; 0 remain |
| Missing JEE (per student) | 253 students, 1,265 rows | cannot be recovered | kept, `jee_missing` flag added |
| One missing OA section | random gaps | solve from known OA total and weights | 50-59 per section |

Order matters: duplicates before text conversion; text conversion before range checks; recovery last.

### 5.2 Decisions

- Student-level values are identical across a student's rows, so gaps are filled from other rows, not with a column mean.
- JEE is **not** imputed during cleaning. Imputing on the full dataset would leak test information. It is imputed inside the model pipeline using training folds only.
- `eligible` is not recomputed from raw JEE, because it was set from the true value.
- Random gaps in `technical` and `hr` are left for pipeline imputation.

### 5.3 Validation

Cleaned values were compared with the answer key: 0 wrong values in tenth_pct, twelfth_pct, cgpa, jee_percentile; OA section max error 0.0; 96 selected rows preserved.

**Lesson learned:** my first validation passed because I accidentally loaded the truth file, so the cleaner had nothing to clean. A test that cannot fail proves nothing. After regenerating the raw file, the log matched the injected mess almost one for one.

## 6. Exploratory Data Analysis

Six charts (`visuals/01-06`), each filtered to rows that reached the stage; histograms use one row per student.

| # | Chart | Insight |
|---|---|---|
| 1 | Histograms of 10th, 12th, JEE, CGPA | [FILL: shape, ceilings at 99 and 10, 253 missing JEE] |
| 2 | Correlation heatmap (eligible rows) | [FILL: which academic column correlates most with OA sections; correlations are lower than in the full population because eligibility restricts the range] |
| 3 | CGPA vs OA total | [FILL: upward but noisy trend; students with the same CGPA fall on both sides of the cutoff, which is why ML has something to predict] |
| 4 | OA sections, cleared vs not | [FILL: which section separates the groups most; coding has the largest weight] |
| 5 | Funnel by category (log scale) | [FILL: stage pass rates; 64 non-dream and 32 dream selected] |
| 6 | Interview scores, selected vs not | [FILL: final_score separates almost perfectly because selection is defined by it, so it cannot be a feature] |

## 7. Feature Selection

**Excluded as features** (documented in `src/features.py`):

| Column | Reason |
|---|---|
| final_score | Used to choose selected candidates (direct leakage) |
| oa_total | Weighted sum of the five sections (redundant) |
| oa_cleared, eligible | Outcomes of earlier stages |
| selected | Target |
| IDs, company_id, company_category | Labels, not student attributes (student_id used only for grouped splits) |

**Kept:** the four academics, `jee_missing`, five OA sections, technical, HR.

**Evidence:** correlation and mutual information of each candidate feature with `selected`, and a feature-feature correlation heatmap (charts 08-10). [FILL: top and bottom features; whether correlation and mutual information agree; 10th and 12th correlate about 0.6 but were both kept because they carry separate weights in the simulator.]

**Principle:** the feature set depends on what is known at prediction time. This gives four sets:

| Set | Features |
|---|---|
| A | Academics (+ jee_missing) |
| B | A + five OA sections |
| C | A + technical, HR |
| D | A + OA sections + technical, HR |

## 8. Modelling

- **Rows:** 960 interviewed applications; target `selected` (96 positive, 10%).
- **Models:** logistic regression (with `StandardScaler`) and random forest, with `class_weight` balancing. Regression: linear and random forest against a `DummyRegressor` baseline.
- **Validation:** 5-fold cross-validation grouped by `student_id` (stratified for classification), because a student appears in several rows and a plain split would leak.
- **Imputation inside a Pipeline** (median), fitted on training folds only.
- **Scaling:** logistic and linear regression use coefficients and a regularisation penalty, so features must share a scale. Tree models only use the order of values and do not need it.
- **Metrics:** ROC-AUC, average precision, recall (accuracy is misleading at a 10% positive rate, since "always no" gives 90%). MAE, RMSE, R² for regression.

## 9. Results

### 9.1 Selection classification (AUC, mean ± SD over folds)

| Features | Logistic AUC | Avg precision | Recall | Random forest AUC |
|---|---|---|---|---|
| A: academics | 0.793 ± 0.029 | 0.350 | 0.696 | 0.754 ± 0.022 |
| B: + OA | 0.900 ± 0.029 | 0.560 | 0.842 | 0.886 ± 0.029 |
| C: + interviews | 0.969 ± 0.009 | 0.855 | 0.906 | 0.966 ± 0.012 |
| D: all | 0.987 ± 0.008 | 0.923 | 0.948 | 0.981 ± 0.007 |

Recall is boosted partly by class balancing, which trades precision. Average precision is the fairer summary. Each fold has only about 19 positives, so numbers are noisy.

### 9.2 Regression (5-fold grouped CV)

| Task | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| OA total from academics | Dummy | 7.34 | 9.11 | −0.001 |
| | Linear | 4.36 | 5.45 | 0.641 |
| | Random forest | 4.54 | 5.66 | 0.613 |
| Final score from academics + OA | Dummy | 5.16 | 6.38 | −0.007 |
| | Linear | 3.81 | 4.71 | 0.448 |
| | Random forest | 4.02 | 4.99 | 0.383 |

The Stage 1 R² (0.641) matches the generator's in-sample R² (0.643), so there is no overfitting. Stage 2 is capped near 0.45 because 60% of the final score comes from interviews not yet observed.

### 9.3 Backtest at the OA cutoff

Out-of-sample predictions of OA total (grouped CV), thresholded at 65: **precision 0.788, recall 0.709** (896 predicted to clear vs 996 actual). The model is slightly conservative (regression to the mean). Rows of the same student share a prediction, so the effective sample is about 1,000 students.

### 9.4 How the stages connect

| Stage | Known information | Prediction | Quality |
|---|---|---|---|
| 1 | academics | OA score | R² 0.64 |
| 2 | + OA sections | final score | R² 0.45 |
| 2 | + OA sections | selected | AUC 0.90 |
| 3 | + interviews | selected | AUC 0.97-0.99 |

Each new stage adds information and predictions improve. Interview noise limits how early the outcome can be known.

## 10. Hypotheses

### H1: OA scores add more than interview scores: **Rejected**

| Gain over Model A | Logistic | Random forest |
|---|---|---|
| B − A (add OA) | +0.107 | +0.132 |
| C − A (add interviews) | +0.176 | +0.212 |

Interviews add more, by about 0.07-0.08 AUC, roughly 2-3 times the fold SD. This is expected because interviews carry 60% of the final score and selection is defined by it. D is only slightly above C because technical and HR are direct components of the final score. A rejected hypothesis, reported as found.

### H2: Coding matters beyond OA total: **Supported (method validated)**

| Evidence | Result |
|---|---|
| Coefficient (technical ~ oa_total + coding) | 0.152 |
| 95% bootstrap CI (resampling whole students) | [0.069, 0.237], excludes 0 |
| Planted value | 0.25, just outside the interval |
| Permutation importance of coding (random forest) | 0.059 ± 0.023 |

Two independent methods agree. The estimate is below the planted value, likely because of range restriction (only OA-clearers reach interviews) and overlap between coding and OA total. The effect was planted, so this shows the method can detect it. The low held-out R² of the forest (0.147) reflects heavy interview noise and does not weaken H2. Bootstrapping by student is needed because a student's rows are correlated; resampling rows would give an interval that is too narrow.

### H3: Coding tie-break decides seats: **Supported as a frequency**

The last seat was tied on final score at 6 of 12 companies, and in all 6 the coding tie-break decided it (the application-ID fallback was never needed). This depends on rounding final scores to integers; with decimals, ties would nearly vanish. It measures how often the rule mattered, not whether it is fair.

### Planted-weights check

Regression of final score on OA, technical, HR recovered 0.401 / 0.349 / 0.249 against planted 0.40 / 0.35 / 0.25. This validates that the pipeline did not corrupt the structure. It is not a discovery, because the formula was written by me.

## 11. Prediction System

`src/predict.py` holds the logic and `app/streamlit_app.py` only displays it, so the app and notebooks cannot disagree.

- **Rules and models are separate.** `is_eligible` (a rule) decides eligibility; the regressor estimates the OA score only for eligible students, since the model was trained on them.
- **Stage 2 recomputes the OA total and cleared rule.** The classifier was trained only on OA-clearers, so a student below the cutoff is reported as "not cleared" and never passed to it.
- **Missing JEE** is treated as not eligible, because the cutoff cannot be verified on an imputed value.
- **Inputs are validated** (for example, a CGPA of 85 raises an error).
- **Outputs:** expected OA total with a rough clearing chance (normal-error assumption using RMSE 5.45), selection score relative to the 10% base rate, expected final score, and a readiness verdict.

Saved models: `stage1_oa_regressor`, `stage2_selection_classifier`, `stage2_final_score_regressor`, plus `feature_lists.json`.

Streamlit tabs: Before the OA, After the OA, What-if simulator, Model Performance. [FILL: insert screenshots from `report/screenshots/`.]

Interpretation notes: the verdict uses the word **readiness**. `selection_score` is a ranking score, not a probability, because balanced class weights inflate it. The verdict thresholds (40%, 70%, 2×) are uncalibrated choices of mine.

## 12. Why the Results Can Be Trusted (and What They Do Not Show)

1. Predictions come from data the model never saw (grouped cross-validation by student).
2. Models beat a mean baseline by a wide margin.
3. Performance rises A → B → C → D, matching how the funnel works.
4. The method recovers planted weights and effects.
5. At the OA cutoff, precision is 0.79 and recall 0.71.

**What they do not show:** that real placements follow these rules. The system demonstrates that the workflow can learn and predict a placement funnel; real data with the same schema is needed to claim more.

## 13. Real-World Value and Requirements

**Students:** see which cutoffs they miss and by how much, their expected OA score, which academic factors move it (what-if), and where they stand after the OA.
**Placement cells:** identify at-risk students early and find the funnel stage that is the bottleneck for a batch.
**Recruiters:** understand which signals predict later-stage success, as input to humans setting cutoffs and weights.

Before real use: (1) retrain on anonymised real data, (2) re-validate and re-test H1-H3, (3) build company-specific models, (4) run a fairness review, (5) obtain consent and protect privacy, (6) keep a human in the loop.

## 14. Limitations and Ethics

- Synthetic data: results validate the method, not reality.
- The simulator is linear, so linear models are favoured; this is not a general finding.
- Selection is top-8 per company, so labels depend on competing applicants and cross-validation is approximate.
- Selection bias from cutoffs: below-cutoff outcomes are unobserved.
- Small positive class (96 selected, about 19 per fold) makes metrics noisy.
- Repeated students: backtest metrics are over applications, about 1,000 effective students.
- Fairness: 10th %, 12th % and JEE can encode background advantage. The tool shows gaps and must not rank or reject people.
- No causal claims.

## 15. Future Work

Retrain on real anonymised data (100-200 rows from seniors would already help); company-specific models; SHAP explanations; calibrated probabilities and verdict thresholds; extension to other professional domains; fairness and privacy review.

## 16. Conclusion

I built a complete, reproducible ML workflow around a simulated placement funnel: generation, cleaning with validation, EDA, leakage-aware feature selection, grouped cross-validated models, hypothesis testing, and a prediction app. Predictions improve as information accumulates (AUC 0.79 → 0.90 → 0.97+), Stage 1 predicts OA scores with R² 0.64, and the planted structure is recovered. H1 was rejected, H2 supported, H3 supported as a frequency. The main limitation is that the data is synthetic, so the next step is to validate the pipeline on real, anonymised placement data.

## Appendix

- **Decision log:** `docs/decisions_log.md` (every design choice, experiment and mistake, including the rho = 0.95 and re-standardisation experiments, the stale-notebook and wrong-file incidents)
- **Key experiments:** rho = 0.95 made 10th and 12th near-duplicates (multicollinearity); removing re-standardisation lowered CGPA correlations from about 0.33 to 0.27 because a mixing formula assumes unit spread
- **Charts:** `visuals/` (01-10, plus `academics_corr_heatmap.png`)
- **Results tables:** `data/processed/model_comparison.csv`, `regression_results.csv`, `backtest_oa.csv`