# Remove

See what the results are by adding `hcc` and or `mcc`. Note `hcc` is already in live NWP consumer

Let's compare the results of
1. train with none
2. train with `mcc`
3. train with `hcc`
4. train with both

Resutls MAE test set. Should remove HCCT. Later we can investigate if we should add `hcc` or `mcc`

| Horizon | ref   | MCC   | HCC   | HCC + MCC |
|---------|-------|-------|-------|-----------|
| 0       | 0.016 | 0.016 | 0.016 | 0.016     |
| 1       | 0.021 | 0.021 | 0.021 | 0.021     |
| 2       | 0.024 | 0.024 | 0.024 | 0.024     |
| 4       | 0.025 | 0.025 | 0.025 | 0.025     |
| 8       | 0.026 | 0.026 |       |           |
| 16      | 0.025 | 0.025 |       |           |
| 24      | 0.025 | 0.025 |       |           |
| 36      | 0.026 | 0.026 |       |           |
