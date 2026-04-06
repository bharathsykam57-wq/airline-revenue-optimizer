# Modeling Decisions

## Why LightGBM over LSTM/deep learning

**Data characteristics:**
- 216 training rows (monthly aggregates, 6 routes, 3 years)
- 27 engineered tabular features
- No raw sequence data — BTS is pre-aggregated monthly

**LSTMs need:**
- Individual booking event sequences (we don't have this)
- Minimum ~1000+ sequence samples for stable training
- Careful sequence padding and masking for variable lengths

**LightGBM wins because:**
- Native quantile regression (objective="quantile")
- Handles 216 rows without overfitting risk at correct depth
- SHAP values are exact, not approximate
- Training time: seconds, not minutes
- No hyperparameter sensitivity to learning rate schedules
- Interpretable feature importance directly maps to elasticity

**Benchmark result:** (to be filled after training)
LightGBM RMSE vs XGBoost RMSE vs Linear baseline RMSE

## Why quantile regression over point estimates

Point estimates in pricing decisions are dangerous.
If model predicts 150,000 passengers but true demand is 90,000,
optimized price will be set too high — revenue loss.

Quantile regression provides:
- q10: conservative scenario — set floor for risk-aware pricing
- q50: expected scenario — primary pricing signal
- q90: optimistic scenario — upside potential

Prediction interval coverage target: 80% of actuals fall
within [q10, q90]. If coverage < 70% — model is overconfident.

## Why route-specific models over one global model

EDA showed COVID recovery varied 3x between routes:
- JFK-LAX: -74.3% drop, -21.3% still below pre-COVID
- MIA-ORD: -45.8% drop, nearly full recovery

A global model averages these patterns and mispredicts both.
Route-specific models learn each route's demand dynamics
independently. Cost: 6 models instead of 1.
Benefit: correct predictions per route.
