# Extend to 40 hours

The idea was to extend the model to 40 hours but still using 36 hours of NWP. This is because the CEDA data we have only goes to 36 hours at the moment.

Resutls for 37 - 40 hours, training is 2021 and test is 2022. The results are from the test

| Forecast Horizon | MAE test [%] |
|------------------|--------------|
| 37               | 0.021        |             
| 38               | 0.022        |             
| 39               | 0.023        |             
| 40               | 0.025        |             

We also wanted to look at the effect of not using NWP data from that forecat horizon. 
Therefore we restrict the maximum forecast horizon to 30 hours to see the differences. 

| Forecast Horizon | MAE [%] | MAE (30 hours) [%] |
|------------------|---------|--------------------|
| 30               | 0.020   | 0.020              |
| 33               | 0.021   | 0.023              |
| 36               | 0.021   | 0.026              |


