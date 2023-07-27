

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

## Results for cross-validation 2016-2022 mean range and quantile coverage

| Horizon | 0.1     | 0.5     | 0.9     |
|---------|---------|---------|---------|
| 0       | 0.21619 | 0.49991 | 0.72409 |
| 1       | 0.24395 | 0.49761 | 0.69554 |
| 2       | 0.2563  | 0.488   | 0.66742 |
| 3       | 0.25588 | 0.49109 | 0.66645 |
| 4       | 0.25631 | 0.48897 | 0.66709 |
| 5       | 0.25536 | 0.49024 | 0.66989 |
| 6       | 0.25666 | 0.49079 | 0.669   |
| 7       | 0.25537 | 0.48862 | 0.66705 |
| 8       | 0.25444 | 0.48715 | 0.66836 |
| 9       | 0.25524 | 0.48759 | 0.66886 |
| 10      | 0.25443 | 0.48719 | 0.66661 |
| 11      | 0.25302 | 0.48774 | 0.66849 |
| 12      | 0.25294 | 0.4895  | 0.66953 |
| 13      | 0.25381 | 0.48953 | 0.66902 |
| 14      | 0.25379 | 0.48751 | 0.67067 |
| 15      | 0.25463 | 0.49091 | 0.66932 |
| 16      | 0.25517 | 0.48787 | 0.66843 |
| 17      | 0.25394 | 0.48815 | 0.66976 |
| 18      | 0.25636 | 0.48961 | 0.67008 |
| 19      | 0.25522 | 0.48977 | 0.66885 |
| 20      | 0.25505 | 0.48906 | 0.66857 |
| 21      | 0.25536 | 0.49072 | 0.671   |
| 22      | 0.25627 | 0.48986 | 0.66942 |
| 23      | 0.25585 | 0.49085 | 0.67046 |
| 24      | 0.25764 | 0.4914  | 0.67382 |
| 25      | 0.25694 | 0.49036 | 0.66981 |
| 26      | 0.25876 | 0.49247 | 0.67047 |
| 27      | 0.26142 | 0.49424 | 0.67243 |
| 28      | 0.25904 | 0.4941  | 0.6705  |
| 29      | 0.26    | 0.49409 | 0.67048 |
| 30      | 0.25909 | 0.49636 | 0.67121 |
| 31      | 0.26019 | 0.49471 | 0.67071 |
| 32      | 0.25914 | 0.49492 | 0.67213 |
| 33      | 0.26286 | 0.49671 | 0.6728  |
| 34      | 0.26168 | 0.49789 | 0.67299 |
| 35      | 0.26133 | 0.48521 | 0.67172 |
| 36      | 0.26343 | 0.49714 | 0.67256 |


## Quantile Scaling Results

For scaling the production quantiles to be more accurate, from this, we need to scale
the lower quantile by 0.4 to be actually at 10%, and the upper quantile by 1.6 to actually be at 90%.


# Monthly Errors and Percentiles

These plots are from training on the whole 2016-2022 dataset (the production model) and testing on the whole dataset, so will be smaller errors than in production, but does give an idea of how well the models do throughout the year.

Percentiles:

![0_pinball_losses_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/d02fc568-df2d-4989-9fcb-3ef0264a288b)
![1_pinball_losses_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/fded24d7-2536-4153-b3ff-1e7382865acb)
![2_pinball_losses_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/8294efac-8c91-4264-9816-a67092c155d5)
![4_pinball_losses_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/9fbf9199-4bc0-4df7-8678-2e1e9138a276)
![8_pinball_losses_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/0b99f433-0d9c-4c45-8185-bf6c0c91ed32)
![12_pinball_losses_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/f563a760-6a89-4716-9a24-2581d424fce5)
![24_pinball_losses_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/27b92f86-b1da-45a6-a939-a0cd9ca8707e)
![36_pinball_losses_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/7692993b-b208-4351-8a48-cfe8b12e968a)


Monthly MAE:
![0_mae_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/2736fafb-1fe7-45a0-84f6-bb9dfb2dfddc)
![1_mae_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/07232ed6-fceb-47d5-b7f2-e0089869e210)
![2_mae_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/f2c4b156-c388-4417-9548-aad49f8de89b)
![4_mae_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/3267aeb4-228c-4c23-9678-a45e0ad696ed)
![8_mae_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/9714f916-9d2e-401b-a493-710481a2a57a)
![12_mae_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/a5373018-6a1e-4a5a-870f-d49c60af9284)
![24_mae_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/50ab93e5-43ac-474d-98e9-9e85a04d6417)
![36_mae_monthly](https://github.com/openclimatefix/uk-pv-national-xg/assets/7170359/27b3b980-d394-4a5c-bd1f-43460291651f)


