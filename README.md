# National XGBoost PV Model

Repository hosting various models used to predict National PV in the UK using Numerical Weather Prediction (NWP) data in addition to historical PV values. Several model types for processing NWPs are considered including:

- Single point downsampling
- Quadrant (4-point) geo-downsampling
- Pretrained CNN downsampling
- Region-masked downsampling

In addition to methods used for preprocessing and training a model for PV forecast, we also provide a pipeline for live model inference.


## Questions

- How to run training? What scripts to run
- Description of the current model
  - What data does it take in? What NWP channels does it use
  - What is the pre processing
  - What ml model is used?
- What happens if there is missing data?
- Where is the model saved?
- How to run inferences? 