# N Estimators

Lets investigate the effects of `n_estimators` on the results.

| Horizon | 750   | 1000  | 1250  | 1500  |
|---------|-------|-------|-------|-------|
| 0       | 0.016 | 0.010 | 0.009 | 0.009 |
| 1       | 0.021 | 0.016 | 0.014 | 0.014 |
| 2       | 0.024 | 0.018 | 0.017 | 0.016 |
| 4       | 0.025 | 0.020 | 0.019 | 0.018 |
| 8       | 0.026 | 0.020 | 0.019 | 0.019 |
| 16      | 0.025 | 0.020 | 0.018 | 0.018 |
| 24      | 0.025 | 0.020 | 0.019 | 0.018 |
| 36      | 0.026 | 0.021 | 0.019 | 0.019 |

I think it would be sensible to move `n_estimators` up to 1250.
Note that training takes longer with a larger  `n_estimators`
