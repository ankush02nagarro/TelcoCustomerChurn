# Training-data EDA insights

26.5% of training customers churned. Compare against a majority-class baseline; accuracy alone can hide missed churners.

For Contract, Month-to-month has the highest observed churn rate (42.9%, n=2704); Two year has 3.0%. Use contract segments to prioritize retention experiments and evaluate renewal support. Treat small groups cautiously.

Median tenure is 10.0 months for churners and 38.0 for non-churners. Use the difference to assess whether onboarding or longer-term engagement deserves attention; this is not a causal result.

Median monthly charges are 80.00 for churners versus 64.45 for non-churners (dataset currency units). Review perceived value and plan suitability rather than assuming price alone explains churn.

For InternetService, Fiber optic has the highest observed churn rate (41.8%, n=2177); No has 6.9%. Investigate service experience and customer mix in the highest-rate segment before choosing an intervention. Treat small groups cautiously.

For PaymentMethod, Electronic check has the highest observed churn rate (45.6%, n=1654); Credit card (automatic) has 14.7%. Review payment friction and contract mix; the payment method itself may not be the cause. Treat small groups cautiously.

Median accumulated charges are 705.08 for churners and 1679.40 for non-churners. Total charges reflect both tenure and price, so low totals should not automatically be interpreted as low customer value.

Among the three continuous features, tenure has the strongest absolute linear association with churn (r=-0.34). This guides investigation but does not capture all nonlinear relationships or establish causality.