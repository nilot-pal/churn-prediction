# Customer churn prediction

Predicting which telecom customers are about to leave, on the IBM Telco dataset. The modelling is
routine; what this project is actually about is **evaluation** — churn is imbalanced and
cost-asymmetric, so accuracy is close to meaningless and the default 0.5 threshold is arbitrary.

The result worth reading is at the bottom: **which model is best depends on the threshold you
operate at, and the ranking flips.**

📓 [**`notebooks/churn_end_to_end.ipynb`**](notebooks/churn_end_to_end.ipynb) — the whole project
in one notebook: EDA, preprocessing, three models, feature importance, threshold tuning and error
analysis.

## Data

IBM Telco Customer Churn — 7,043 customers, ~27% churn. Contract type, tenure, monthly and total
charges, payment method, and which services each customer subscribes to.

## Preprocessing

Built as a scikit-learn `Pipeline` with a `ColumnTransformer`, so that scaling and encoding are
fitted **inside** cross-validation rather than on the full dataset — otherwise test data leaks
into the fit and every score afterwards is optimistic.

`TotalCharges` contains blanks for customers with zero tenure, which read as strings rather than
NaN. These are coerced to numeric and imputed with `SimpleImputer`.

## Models

Three, all through the same pipeline, split and evaluation protocol:

| Model | ROC AUC | F1 (churn) |
|---|---|---|
| Logistic regression | **0.84** | 0.61 |
| Decision tree | 0.83 | 0.61 |
| Random forest | 0.84 | 0.55 |

At the default threshold, logistic regression wins. Random forest has the highest **accuracy**
(0.79) and the highest **precision** (0.65), but its recall drops to 0.48 — it misses over half
the churners. Logistic regression, with class weighting, gets recall to 0.76 at precision 0.52.

Similar ROC AUC across all three says they rank churn risk about equally well. The differences are
in where each one puts its decision boundary, which is a threshold question, not a model question.

## What drives churn

From random-forest importances, cross-checked against logistic-regression coefficients and
decision-tree impurity:

1. **Tenure** — the strongest single predictor; short-tenure customers churn far more
2. **Month-to-month contract** — one- and two-year contracts push the other way
3. **TotalCharges and MonthlyCharges**
4. **Payment method**

All three models agree on the top features, which is the useful check — a ranking that only one
model reports is usually an artefact of how that model measures importance.

## ⭐ Threshold tuning: the ranking flips

Lowering the decision threshold from 0.5 to 0.3:

| At threshold 0.3 | Precision | Recall | F1 |
|---|---|---|---|
| Logistic regression | 0.41 | **0.93** | lower than at 0.5 |
| Random forest | 0.52 | 0.78 | **0.63** |

Logistic regression catches nearly every churner — 93% recall — but at 41% precision, so most
customers flagged for a retention offer were never going to leave. Random forest absorbs the same
threshold change far better: recall rises to 0.78 while precision holds at 0.52, and its F1 goes
**up** to 0.63, overtaking logistic regression.

**So the model that looked worse at the default threshold is the better one at the operating point
a business would actually choose.** Model selection and threshold selection are one decision, not
two, and evaluating at 0.5 alone would have led to the wrong choice.

Which threshold is right depends on the numbers: customer lifetime value against the cost of a
retention intervention. If retention offers are cheap relative to losing a customer, recall is
worth buying with precision. The notebook works through an example costing.

## Error analysis

The random forest's mistakes are not random. They concentrate in **month-to-month contracts**, at
**moderate-to-high tenure** rather than among new customers, and at a relatively high average
monthly charge (~$73) with wide spread.

That is a coherent group — established customers on flexible contracts paying above-average bills.
Churn in that segment is driven by something the dataset doesn't contain, likely price sensitivity
or a service complaint. More model capacity will not fix it; more features would.

## Running it

```
churn-prediction/
├── data/raw/          # IBM Telco dataset
├── notebooks/
│   └── churn_end_to_end.ipynb
└── README.md
```

Needs `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`. Run the notebook top to bottom.

## Notes

Single notebook rather than a `src/` package. For a project this size, splitting the pipeline
across modules would add indirection without making anything reusable — the notebook keeps each
result next to the reasoning that produced it. A larger project would be organised differently.
