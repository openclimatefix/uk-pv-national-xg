# Add more GSP lags

Currently we use 1 hour, 2 hour and 24 hour lags. Lets add a few more

1. original lags
2. 1,2, 3,6,12,15, 18, 21 and 24 hour lags
3. add all 24 hours lags
4. Just add 3 hour lag


| Horizon | original | 3 hour lags | all 1 hour | just add 3 hour |
|---------|-----|-------------|------------|-----------------|
| 0       |  0.016 | 0.016       | 0.016      | 0.016           |
| 1       |  0.021 | 0.021       | 0.021      | 0.021           |
| 2       | 0.025 | 0.024       | 0.024      | 0.024           |
| 4       |  0.025 | 0.025       | 0.025      | 0.025           |
| 8       |    0.026   | 0.026 | 0.026      | 0.026           |
| 16      |    0.025     | 0.025| 0.025      | 0.025           |
| 24      |  0.025     | 0.025 | 0.025      | 0.025           |
| 36      |  0.026     | 0.026 | 0.026      | 0.026           |

So its slightly better to add the 3 hour too, but it doesn't make too much difference. 