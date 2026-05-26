# Methodology Notes — Employee Attrition Prediction

## Model Selection

A **logistic regression** classifier was selected over alternative algorithms (e.g., Support Vector Machines) for two primary reasons:

**Scalability:** Logistic regression handles large datasets efficiently. SVMs, while mathematically powerful, become computationally expensive as dataset size grows — making logistic regression the more practical choice for production-scale HR datasets.

**Interpretability:** Logistic regression coefficients directly quantify each variable's influence on the predicted outcome. This makes the model's findings immediately actionable for non-technical stakeholders such as HR managers and department leads — a capability that kernel-based SVM approaches cannot easily provide.

---

## Feature Selection Philosophy

The full dataset contains 28 potential predictor variables. Rather than using all 28, a subset of 10 was selected based on domain knowledge and theoretical relevance to attrition behavior. This approach keeps the model computationally efficient, interpretable, and focused on the variables most likely to drive actionable decisions.

| Feature | Rationale |
|---------|-----------|
| `Age` | Correlates with career stage and likelihood of seeking new opportunities |
| `DistanceFromHome` | Longer commutes negatively impact work-life balance and increase attrition risk |
| `EnvironmentSatisfaction` | Dissatisfaction with the work environment is a known driver of voluntary turnover |
| `JobSatisfaction` | Higher job satisfaction is consistently associated with lower attrition |
| `WorkLifeBalance` | Poor work-life balance is one of the most cited reasons for leaving a company |
| `MonthlyIncome` | Financial dissatisfaction amplifies other attrition drivers |
| `YearsAtCompany` | Shorter tenure employees are statistically more likely to leave |
| `YearsSinceLastPromotion` | Stagnation without advancement is a strong predictor of disengagement |
| `YearsWithCurrManager` | Strong manager relationships are a key retention factor |
| `PercentSalaryHike` | Recent salary growth signals recognition and reduces attrition likelihood |

After iterative testing on the balanced dataset, the model was further refined to the **4 highest-signal features:**
`Age`, `WorkLifeBalance`, `YearsAtCompany`, `YearsSinceLastPromotion`

Notably, removing `MonthlyIncome` improved balanced accuracy — suggesting it introduced noise rather than meaningful signal in this particular dataset configuration.

---

## Adaptability

While built for employee attrition, this modeling framework is intentionally general. The same pipeline — binary target encoding, class balancing, feature selection, and logistic regression — can be adapted to related classification problems such as:

- Customer churn prediction
- Patient readmission risk
- Loan default likelihood
- Student dropout risk

The interpretability of logistic regression coefficients makes it especially well-suited for any domain where stakeholders need to understand *why* a prediction was made, not just *what* it is.
