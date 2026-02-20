# ✈️ Airbnb New User Bookings — Destination Prediction

> **Kaggle Competition:** [Airbnb New User Bookings](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings)  
> **Task:** Predict the first booking destination of new Airbnb users across 12 classes  
> **Metric:** NDCG@5 (Normalised Discounted Cumulative Gain at rank 5)  
> **Final Validation NDCG@5:** `0.8651`

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Summary](#pipeline-summary)
- [Results](#results)
- [Key Findings](#key-findings)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Report](#report)

---

## Overview

This project develops a machine learning pipeline to predict which country a new Airbnb user will make their first booking in, using demographic data, session logs, and signup behaviour. The 12 target classes are:

`NDF` · `US` · `other` · `FR` · `IT` · `GB` · `ES` · `CA` · `DE` · `NL` · `AU` · `PT`

The pipeline progresses from a Logistic Regression baseline (NDCG@5 = 0.42) to a three-stage hierarchical ensemble (NDCG@5 = **0.8651**) — a total gain of **+0.44**.

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `train_users_2.csv` | 213,451 | Training users with labels |
| `test_users.csv` | 62,096 | Test users (no labels) |
| `sessions.csv` | ~10.5M | Web session logs per user |
| `countries.csv` | 10 | Country-level statistics |
| `age_gender_bkts.csv` | — | Age/gender distribution buckets |

> Data is not included in this repository. Download it from the [Kaggle competition page](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data).

**Class distribution (training set):**

```
NDF    58.35%    US     29.22%    other   4.73%
FR      2.35%    IT      1.33%    GB      1.09%
ES      1.05%    CA      0.67%    DE      0.50%
NL      0.36%    AU      0.25%    PT      0.10%
```

---

## Project Structure

```
airbnb-new-user-bookings/
│
├── Airbnb_0_0_Editing.ipynb      # Main notebook (full pipeline)
│
├── data/                          # Place Kaggle data files here
│   ├── train_users_2.csv
│   ├── test_users.csv
│   ├── sessions.csv
│   ├── countries.csv
│   └── age_gender_bkts.csv
│
├── report/                        # LaTeX report source
│   ├── main.tex
│   ├── chapters/
│   │   ├── chapter1_introduction.tex
│   │   ├── chapter2_dataset.tex
│   │   ├── chapter3_eda.tex
│   │   ├── chapter4_preprocessing.tex
│   │   ├── chapter5_features.tex
│   │   ├── chapter6_merging.tex
│   │   ├── chapter7_modeling.tex
│   │   ├── chapter8_hierarchical.tex
│   │   ├── chapter9_imbalance.tex
│   │   ├── chapter10_ensemble.tex
│   │   ├── chapter11_crossvalidation.tex
│   │   ├── chapter12_evaluation.tex
│   │   ├── chapter13_feature_importance.tex
│   │   └── chapters14_17_final.tex
│   └── figures/
│
└── README.md
```

---

## Pipeline Summary

### Phase 1 — Data Preprocessing
- Sessions: dropped null `user_id` rows (34,496), imputed `action_type` / `action_detail` via two-pass strategy, imputed `secs_elapsed` with median
- Users: corrected 828 impossible age entries (year-formatted), capped ages > 90, imputed 116,866 nulls with median (33), parsed 10 temporal features from registration dates
- Categorical grouping: applied 2,000-count cutoff to reduce cardinality across 13 features

### Phase 2 — Feature Engineering
Session aggregations per user (23 features):
- Action counts, most-frequent action values, unique value counts
- 9 duration statistics from `secs_elapsed` (mean, median, min, max, sum, 3 pause types)
- 4 device category binary flags
- `age_group` ordinal bucket (created pre-imputation — 2nd highest permutation importance)

### Phase 3 — Merging & Encoding
- Left join: sessions onto users (140,064 sessionless users retained to avoid selection bias)
- One-hot encoding: 13 categorical features → ~100 binary columns
- Min-Max scaling: all continuous features normalised to [0, 1]
- Final feature matrix: **275,547 × 132**, zero nulls

### Phase 4 — Modeling (7 configurations)

| Model | NDCG@5 |
|-------|--------|
| Logistic Regression | 0.4215 |
| Decision Tree | 0.4471 |
| Linear SVM | 0.5858 |
| LightGBM (GPU) | 0.6997 |
| XGBoost (GPU) | 0.8301 |
| CatBoost (GPU, default) | 0.8304 |
| CatBoost (Optuna-tuned) | 0.8307 |

### Phase 5 — Hierarchical Model
Three-stage CatBoost decomposition:
```
Stage A: All users       →  NDF vs Trip          (binary Logloss,    600 iterations)
Stage B: Bookers only    →  US vs International   (binary Logloss,    600 iterations)
Stage C: Intl bookers    →  10-class destination  (MultiClass Logloss, 800 iterations)
```
Probabilities combined via the conditional chain:
- `P(NDF) = 1 − P(Trip)`
- `P(US) = P(Trip) × P(US | Trip)`
- `P(country_k) = P(Trip) × P(Intl | Trip) × P(country_k | Intl)`

**Hierarchical NDCG@5: 0.8610** (+0.0306 over best flat model)

### Phase 6 — Imbalance Experiments

| Strategy | NDCG@5 | vs Baseline |
|----------|--------|-------------|
| CatBoost (default) | 0.8304 | — |
| Hybrid Sampling (RUS + SMOTE) | 0.8269 | −0.0035 |
| Balanced Bootstrap (15k/class) | 0.8041 | −0.0263 |

> Resampling consistently hurts NDCG@5 on this dataset — resampling was excluded from the final pipeline.

### Phase 7 — Ensemble & Validation

**2-Model Ensemble** (Hierarchical × 0.70 + CatBoost Optuna × 0.30):

| Metric | Value |
|--------|-------|
| **NDCG@5** | **0.8651** |
| Accuracy | 0.7422 |
| Macro F1 | 0.1233 |
| Weighted F1 | 0.6817 |
| Macro Recall | 0.1291 |
| Top-5 Accuracy | 0.9586 |

**5-Fold Cross-Validation** (CatBoost Optuna + XGBoost ensemble):
```
Mean NDCG@5 = 0.8305   Std = 0.0002   Range = 0.0007
```

---

## Results

**Full model leaderboard (sorted by NDCG@5):**

| Rank | Model | NDCG@5 |
|------|-------|--------|
| 🥇 1 | Ensemble 2-Model (Hier. + CatBoost Optuna) | **0.8651** |
| 🥈 2 | Hierarchical CatBoost (3-stage) | 0.8610 |
| 🥉 3 | Ensemble 3-Model (Hier. + CatBoost + XGB) | 0.8595 |
| 4 | CatBoost (Optuna-tuned) | 0.8307 |
| 5 | CatBoost (GPU, default) | 0.8304 |
| 6 | XGBoost (GPU) | 0.8301 |
| 7 | CatBoost (Hybrid Sampling) | 0.8269 |
| 8 | CatBoost (Balanced Bootstrap) | 0.8041 |
| 9 | LightGBM (GPU) | 0.6997 |
| 10 | Linear SVM | 0.5858 |
| 11 | Decision Tree | 0.4471 |
| 12 | Logistic Regression | 0.4215 |

---

## Key Findings

- **Architecture beats tuning.** Optuna hyperparameter search over 20 trials gained +0.0003 NDCG@5. The hierarchical decomposition alone gained +0.0306 — 100× more.
- **Gradient boosting is non-negotiable.** No linear or shallow model exceeded NDCG@5 = 0.59. The jump from Linear SVM to LightGBM is +0.11.
- **Resampling hurts NDCG@5.** Forced class balance shifts probability mass away from the dominant classes that the model predicts most confidently, degrading ranking quality.
- **Age is the #1 feature** across every model and importance method — by a wide margin. Combined with temporal registration signals and session duration statistics, these three feature groups carry the majority of predictive signal.
- **The pipeline is highly stable.** CV std = 0.0002 NDCG@5 across 5 folds confirms genuine generalisation.

---

## Requirements

```bash
pip install numpy pandas scikit-learn catboost xgboost lightgbm imbalanced-learn optuna matplotlib seaborn
```

GPU support is required for CatBoost, XGBoost, and LightGBM GPU configurations. The notebook was developed on a Windows machine with CUDA. CPU fallback is possible by changing:
- `task_type="GPU"` → `task_type="CPU"` (CatBoost)
- `tree_method="gpu_hist"` → `tree_method="hist"` (XGBoost)
- `device="gpu"` → `device="cpu"` (LightGBM)

**Python version:** 3.10+

---

## How to Run

1. Clone this repository
```bash
git clone https://github.com/<your-username>/airbnb-new-user-bookings.git
cd airbnb-new-user-bookings
```

2. Download the data from Kaggle and place CSV files in `data/`
```bash
kaggle competitions download -c airbnb-recruiting-new-user-bookings
unzip airbnb-recruiting-new-user-bookings.zip -d data/
```

3. Open and run the notebook
```bash
jupyter notebook Airbnb_0_0_Editing.ipynb
```

> Run all cells sequentially. The full pipeline (including GPU training) takes approximately 15–25 minutes end-to-end.

---

## Report

A full 17-chapter academic report documenting every step of the pipeline — from EDA through ensemble learning — is available in the `report/` directory. The report was compiled in LaTeX (Overleaf-compatible) and covers:

| Chapters | Topic |
|----------|-------|
| 1–3 | Introduction, Dataset, EDA |
| 4–6 | Preprocessing, Feature Engineering, Merging |
| 7–10 | Modeling, Hierarchical Strategy, Imbalance Handling, Ensemble |
| 11–13 | Cross-Validation, Model Comparison, Feature Importance |
| 14–17 | Final Model, Discussion, Future Work, Conclusion |

---

## Author

Built as part of a machine learning course project.  
Competition: [Airbnb New User Bookings — Kaggle (2015)](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings)
