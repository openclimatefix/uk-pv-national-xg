# Remove

In live the NWP conusmer only has `hcc` not `hcct` so we need to remove this from the config

Let's compare the results of
1. training with `hcct` and running inference
2. training with `hcct` and running inference with no `hcct` data
3. training with out `hcct`

Resutls MAE test set. Should remove HCCT. Later we can investigate if we should add `hcc` or `mcc`

| Horizon | HCCT  | HCCT not in inference | no HCCT |
|---------|-------|-----------------------|---------|
| 0       | 0.016 | 0.019                 | 0.016   |
| 1       | 0.021 | 0.027                 | 0.021   |
| 2       | 0.024 | 0.032                 | 0.024   |
| 4       | 0.025 | 0.031                 | 0.025   |
| 8       | 0.026 | 0.030                 | 0.026   |
| 16      | 0.025 | 0.029                 | 0.025   |
| 24      | 0.025 | 0.029                 | 0.025   |
| 36      | 0.026 | 0.029                 | 0.026   |
