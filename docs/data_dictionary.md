# Data dictionary

## Source

- **Dataset:** IBM Telco Customer Churn
- **Public source:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Local file:** `data/raw/telco_churn.csv`
- **Rows:** 7,043
- **Raw columns:** 21
- **Target:** `Churn`
- **Observed churn rate:** 26.54% (1,869 churners)

The file is tracked unchanged in the repository. Its SHA-256 hash matches the original `data.csv` that shipped with the legacy notebook.

## Raw fields

| Field | Type | Allowed values / notes |
| --- | --- | --- |
| `customerID` | string | Unique customer identifier. Not a model feature. |
| `gender` | categorical | `Female`, `Male` |
| `SeniorCitizen` | categorical | Source uses `0` / `1`; cleaned to `No` / `Yes` |
| `Partner` | categorical | `No`, `Yes` |
| `Dependents` | categorical | `No`, `Yes` |
| `tenure` | integer | Months with the company, `0`–`72` |
| `PhoneService` | categorical | `No`, `Yes` |
| `MultipleLines` | categorical | `No phone service`, `No`, `Yes` |
| `InternetService` | categorical | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | categorical | `No`, `Yes`, `No internet service` |
| `OnlineBackup` | categorical | `No`, `Yes`, `No internet service` |
| `DeviceProtection` | categorical | `No`, `Yes`, `No internet service` |
| `TechSupport` | categorical | `No`, `Yes`, `No internet service` |
| `StreamingTV` | categorical | `No`, `Yes`, `No internet service` |
| `StreamingMovies` | categorical | `No`, `Yes`, `No internet service` |
| `Contract` | categorical | `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling` | categorical | `No`, `Yes` |
| `PaymentMethod` | categorical | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `MonthlyCharges` | float | Current monthly charge in USD |
| `TotalCharges` | float / string | Lifetime charges in USD. The source has 11 blank strings; all have `tenure=0`. |
| `Churn` | categorical target | `No`, `Yes`; mapped to `0`, `1` for training |

## Cleaning rules

1. `customerID` is stripped and validated for uniqueness.
2. `SeniorCitizen` is normalised from `0/1` to `No/Yes`.
3. `tenure` and `MonthlyCharges` must be numeric.
4. `TotalCharges` is coerced to numeric. Blanks are filled as `MonthlyCharges × tenure`; for a new customer with `tenure=0`, that is zero.
5. `Churn` must contain only `Yes` / `No` in labelled data.
6. The resulting feature frame must have no missing values.

## Engineered features

| Feature | Definition | Why it exists |
| --- | --- | --- |
| `AvgMonthlySpend` | `TotalCharges / tenure`; falls back to `MonthlyCharges` when `tenure=0` | Normalises lifetime spend for tenure and captures price level. |
| `ServiceCount` | Count of the eight service columns equal to `Yes`: phone, multiple lines, online security, online backup, device protection, tech support, streaming TV, streaming movies | Summarises service depth and bundle stickiness. |
| `IsNewCustomer` | `1` when `tenure <= 6`, else `0` | Makes the high-risk onboarding window explicit. |

## Model feature list

The model uses 22 features:

```text
tenure
MonthlyCharges
TotalCharges
AvgMonthlySpend
ServiceCount
IsNewCustomer
gender
SeniorCitizen
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
```

Categorical values are one-hot encoded with unseen categories ignored. Numeric values are median-imputed defensively and standardized for the logistic regression.

## Inference schema

For batch scoring, provide the 19 raw feature columns; `customerID` is optional and `Churn` is ignored if present. See the table above. Generated IDs use the form `row-1`, `row-2`, and so on.

## Derived evaluation views

| Artifact | Description |
| --- | --- |
| `reports/evaluation/test_predictions.csv` | Holdout customer IDs, probabilities, flags, risk tiers, observed churn, and revenue-at-risk estimates. |
| `reports/evaluation/risk_deciles.csv` | Holdout customers grouped into probability deciles with churn rate, cumulative capture, and expected annual revenue at risk. |
| `reports/evaluation/segment_*.csv` | Observed churn, customer counts, and revenue loss by contract, internet service, payment method, and tenure band. |
| `models/feature_importance.csv` | Permutation importance on the holdout set. |

## Privacy and provenance

The dataset is public and anonymised; it contains no names, addresses, or payment credentials. `customerID` is an opaque identifier. Still, treat customer-level scores as sensitive business data and limit access to retention teams with a legitimate purpose.
