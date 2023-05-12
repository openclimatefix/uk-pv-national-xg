# Probablistic Forecasts with Quantile Regression

Lets investigate the effects of quantil regression on the results. The deciles of 10% and 90% are also forecasted.
Additionally, the MAE of the median is calculated as well. This is the Pinball loss from scipy for the quantiles. 
This is supposed to be the same as MAE when the quantile is 0.5, but seems to be slightly different for this case, about half the level of 
our MAE for the median.

| Horizon | 0.1   | 0.5   | 0.9   | MAE 0.5 | Prev MAE 0.5 |
|---------|-------|-------|-------|---------|---------|
| 0       | 0.00261 | 0.00444 | 0.00245 | 0.009   | 0.009 |
| 1       | 0.0041 | 0.00686 | 0.00377 | 0.014   | 0.014 |
| 2       | 0.00526 | 0.00858 | 0.00471 | 0.017   | 0.017 |
| 4       | 0.00543 | 0.00879 | 0.00486 | 0.018   | 0.019 |
| 8       | 0.0054 | 0.00897 | 0.00531 | 0.018   | 0.019 |
| 16      | 0.00547 | 0.00886 | 0.00509 | 0.018   | 0.018 |
| 24      | 0.00518 | 0.00886 | 0.00525 | 0.018   | 0.019 |
| 36      | 0.00526 | 0.00929 | 0.00556 | 0.019   | 0.019 |

Counts below the quantiles. These show the number of times the actual value was below the quantile.
For 10th percentile, it should be around 10%, and for 90% it should be around 90%. It seems that the model
is not necessarily fully calibrated. This is also done for non-night time (defined as true generation above 1% of the capacity), and
for all times, including night time.

Fraction of actual generation below the percentile for all times:

| Horizon | 0.1   | 0.5      | 0.9      |
|---------|-------|----------|----------|
| 0       | 0.60806 | 0.77385  | 0.88295  |
| 1       | 0.62665 | 0.78033  | 0.88061  |
| 2       | 0.63707 | 0.78428  | 0.87586  |
| 4       | 0.64015 | 0.78388  | 0.87372  |
| 8       | 0.63601 | 0.77732  | 0.86664  |
| 16      | 0.63801 | 0.77238    | 0.86396  |
| 24      | 0.6328 | 0.77144    | 0.86811  |
| 36      | 0.63687 | 0.77539   | 0.86797  |

Fraction of actual generation below the percentile for only daylight (Generation > 0.01) times:

| Horizon | 0.1   | 0.5   | 0.9   |
|---------|-------|-------|-------|
| 0       | 0.2478 | 0.56319| 0.769 |
| 1       | 0.28589| 0.57784 | 0.7627 |
| 2       | 0.30172 | 0.58307 | 0.75127 |
| 4       | 0.30902 | 0.58279 | 0.74783 |
| 8       | 0.30172 | 0.56999 | 0.73242 |
| 16      | 0.30557 | 0.5586 | 0.72808 |
| 24      | 0.29246 | 0.55904 | 0.73713 |
| 36      | 0.29982 | 0.56576 | 0.73557 |