# NHS-TT-Baseline-Subgroup-Evaluation-and-Recalibration

This repository contains the analysis code for a study evaluating previously developed baseline prediction models in **NHS Talking Therapies**. The project utilises an independent evaluation framework and post-processing calibration updates to assess model fairness and accuracy across diverse patient groups.

## 🔗 Related Repositories
This work is designed to be used alongside the following papers:
* **Paper 1 (Baseline Development):** [NHS-TT-Baseline-Prediction-Models](https://github.com/nourkanso/NHS-TT-Baseline-Prediction-Models)
* **Paper 2 (Dynamic Models):** [NHS-TT-Dynamic-Prediction-Models](https://github.com/nourkanso/NHS-TT-Dynamic-Prediction-Models)

---

## Overview
This paper does not redevelop the baseline models. Instead, it loads the fixed models produced in Paper 1 and evaluates:
1.  **Baseline predictions**
2.  **Global logistic recalibration**
3.  **Subgroup-specific logistic recalibration**
4.  **Proportional Multicalibration (PMC)**

### Data Source and Cohort
* **Timeline:** January 1, 2018 – August 27, 2024.
* **Criteria:** Adult patients (≥18) receiving high-intensity NHS Talking Therapies (3–21 attended sessions).
* **Note:** This repository **does not** include raw CRIS data due to privacy and governance restrictions.
  
---

## Outcomes & Subgroups
The models predict three standard NHS Talking Therapies outcomes at the final therapy session for both depression (**PHQ-9**) and anxiety (**GAD-7**):
* **Reliable Improvement** (Full treated sample)
* **Recovery** (Condition-specific subsample)
* **Reliable Recovery** (Condition-specific subsample)

### Subgroups Evaluated
| Category | Subgroups |
| :--- | :--- |
| **Gender** | Woman, Man |
| **Sexual Orientation** | Straight, Lesbian/Gay, Bisexual |
| **Ethnicity** | White, Black, Asian, Mixed, Other |
| **Employment** | Employed, Unemployed, Long-term sick/disabled, Student, Retired |
| **Intersectional** | Gender × each attribute listed above |

---

## ⚙️ Evaluation & Recalibration Design
The study employs a **global 70/30 train-test split**:
* **70% Training Set:** Used for global recalibration fitting, subgroup recalibration fitting, and PMC tuning.
* **30% Test Set:** Held out for final evaluation of all approaches.

### Recalibration Procedures
1.  **Global Logistic Recalibration:** A model fitted on the global training set using the logit of the baseline predicted risk.
2.  **Subgroup-specific Logistic Recalibration:** Separate models fitted within each subgroup using subgroup members in the training set.
3.  **Proportional Multicalibration (PMC):** Applied via `PMCBoost` (La Cava, 2024). This iterative method enforces calibration constraints across single-attribute and intersectional indicators.

---

## 📈 Performance Metrics
Evaluation is conducted on the held-out test set using:
* **AUC** (Discrimination)
* **Brier Score** (Overall calibration accuracy)
* **Logit-based intercept and slope** (Calibration components)
* **Quantile-binned ECE** (Calibration error using quantile bins)


