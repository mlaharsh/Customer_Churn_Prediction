# Model card — Churn Signal

## Model details

| | |
| --- | --- |
| **Project** | Churn Signal — customer churn prediction and retention workbench |
| **Model version** | 1.0.0 |
| **Model type** | Binary classification; logistic regression with one-hot encoded categorical features and standardized numeric features |
| **Artifact** | `models/churn_model.joblib` |
| **Training date** | 2026-10-26 |
| **Owner** | MLA Harsh ([@mlaharsh](https://github.com/mlaharsh)) |
| **License** | MIT |
| **Dataset** | IBM Telco Customer Churn (public Kaggle dataset) |
| **Source code** | https://github.com/mlaharsh/Customer_Churn_Prediction |

The deployed bundle contains the fitted pipeline, the selected decision threshold, training-set baselines for one-at-a-time explanations, the ordered feature list, risk-band definitions, and a snapshot of the holdout metrics.

## Intended use

**Primary use.** Prioritise retention outreach and size a campaign for a telecom customer base. The model returns a churn probability, a risk tier, and an expected revenue-at-risk estimate. A retention team can combine that ranking with account facts and capacity constraints.

**Intended users.** Data analysts, retention operations teams, and portfolio reviewers evaluating the reproducibility and decision quality of the project.

**Out of scope.**

- Causal claims about what will save a customer. This is a risk model, not an uplift model.
- Individual pricing, credit, eligibility, or service-denial decisions.
- Replacing human judgment in a customer conversation.
- Production use on a materially different population without validation and monitoring.

## Training data

| | |
| --- | --- |
| Rows | 7,043 customers |
| Raw columns | 21 |
| Target | `Churn` (`Yes` → 1, `No` → 0) |
| Observed churn rate | 26.54% (1,869 churners) |
| Split | 80% train / 20% stratified holdout, `random_state=42` |
| Missing values | 11 blank `TotalCharges` records, all with `tenure=0`; filled as `MonthlyCharges × tenure` (zero for these records) |
| Duplicate IDs | None |
| Average tenure | 32.37 months |
| Average monthly charge | $64.76 |

The source is a widely used, anonymised historical Telco dataset. It does not include a timestamp suitable for temporal validation. The model should be revalidated on current data before operational use.

### Features

The model uses 22 features:

- Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`, `AvgMonthlySpend`, `ServiceCount`, `IsNewCustomer`.
- Categorical: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.

Three derived features are added deterministically:

- `AvgMonthlySpend = TotalCharges / tenure`, or `MonthlyCharges` when `tenure=0`.
- `ServiceCount` counts the eight service columns equal to `Yes`.
- `IsNewCustomer = 1` when `tenure <= 6`.

`customerID` is an identifier and is not a model feature. `Churn` is the target and is never a feature.

## Model selection and evaluation

Five candidate pipelines were compared with 5-fold stratified cross-validation on the training split:

| Candidate | CV ROC-AUC | Holdout ROC-AUC | Holdout F1 @ 0.5 |
| --- | ---: | ---: | ---: |
| **Logistic regression (selected)** | **0.8474** | **0.8457** | 0.6175 |
| Gradient boosting | 0.8456 | 0.8438 | 0.5904 |
| Random forest | 0.8429 | 0.8376 | 0.6135 |
| Histogram gradient boosting | 0.8313 | 0.8288 | 0.5690 |
| Extra trees | 0.8284 | 0.8253 | 0.5868 |

The selection rule was **highest CV ROC-AUC**. The selected model was evaluated once on the stratified holdout split (1,409 customers).

### Holdout metrics at the deployed threshold (0.583)

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7686 |
| Precision | 0.5488 |
| Recall | 0.7219 |
| F1 | 0.6236 |
| ROC-AUC | 0.8457 |
| PR-AUC | 0.6532 |
| Brier score | 0.1657 |
| Log loss | 0.4909 |
| Specificity | 0.7855 |
| Negative predictive value | 0.8866 |
| Alert rate | 0.3492 |
| Top-decile churn capture | 28.61% |
| Top-decile lift | 2.86× |

Confusion matrix at 0.583:

| | Predicted retain | Predicted churn |
| --- | ---: | ---: |
| Actually retained | 813 | 222 |
| Actually churned | 104 | 270 |

### Threshold policy

The threshold is not 0.5. It was selected to maximise F1 on **out-of-fold training predictions**, not on the holdout set. That means the holdout evaluation remains a cleaner estimate of the selected pipeline's ranking and a reasonable, if still retrospective, estimate of the threshold's behavior.

A real campaign should choose the threshold from contact capacity and the relative cost of a missed churner versus a wasted contact. The app exposes a slider for this reason.

## Explainability

The app provides a one-at-a-time sensitivity check: each feature is reset to its training-set median or mode while the others stay fixed. The change in predicted probability is shown in percentage points.

This is **not** SHAP, a causal effect, or a counterfactual guarantee. It is a transparent diagnostic that helps a retention agent ask better questions. Permutation importance on the holdout set identifies the features whose shuffling most reduces ROC-AUC: tenure, internet service, contract, monthly charges, and the new-customer flag carry most of the signal.

## Limitations

1. **Retrospective holdout.** No temporal validation. The dataset is historical and may not represent the current customer mix.
2. **No causal uplift.** The model does not estimate whether a retention offer changes behavior.
3. **Correlated features.** Permutation importance can split or hide credit among correlated variables such as tenure, total charges, and average monthly spend.
4. **Calibration.** Brier 0.1657 is acceptable for ranking and campaign sizing but probabilities are not exact individual odds.
5. **Fairness is not audited.** Gender and senior-citizen status are in the schema. A production use should evaluate subgroup performance, error rates, and business impact, and remove features that create unjustified disparate treatment.
6. **Distribution shift.** Pricing, product mix, and customer behavior can drift. Monitor score distributions, alert rates, calibration, and realized churn after deployment.
7. **Single dataset.** Results should not be generalized to another telecom provider without retraining and validation.

## Ethical and operational considerations

- Use the score to **prioritise human outreach**, not to deny service or set individual prices.
- Keep a human in the loop for customer-facing decisions.
- Do not expose customer-level scores to people without a legitimate retention purpose.
- Track outcomes for contacted and non-contacted customers so the team can distinguish model risk from intervention effect.
- Record the threshold, model version, and date alongside every exported campaign list.

## Deployment and monitoring

**Deployment.** The native Streamlit app reads the committed bundle and evaluation artifacts; no secrets are required. The public GitHub Pages scorer is a client-side implementation of the same exported pipeline, generated by `python -m churn.browser_export`; it has no backend or persistence and is a portfolio demo, not a production decision endpoint.

**Monitoring checklist.**

- Score distribution and alert rate by week.
- Calibration and Brier score on labelled outcomes when they arrive.
- Recall at the operating threshold for the highest-value segments.
- Drift in tenure, contract mix, monthly charges, and service adoption.
- Subgroup performance and complaint / opt-out rates.
- Campaign save rate versus a matched control or holdout.

Retrain with `python -m churn.train` when the data or decision policy changes, review the generated `reports/evaluation/summary.md`, and commit the new artifacts together.

## Reproduction

```bash
git clone https://github.com/mlaharsh/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m churn.train
python -m pytest -q
streamlit run app.py
```

## Citation and source

- Dataset: [IBM Telco Customer Churn, Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Evaluation detail: [`reports/evaluation/summary.md`](../reports/evaluation/summary.md)
- Metrics artifact: [`models/metrics.json`](../models/metrics.json)
- Data dictionary: [`docs/data_dictionary.md`](data_dictionary.md)

This model card follows the spirit of standard model documentation: intended use, training data, evaluation, limitations, and monitoring. It is a portfolio artifact, not a regulatory validation package.
