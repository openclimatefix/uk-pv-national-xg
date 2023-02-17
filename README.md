# National XGBoost PV Model

Repository hosting various models used to predict National PV in the UK using Numerical Weather Prediction (NWP) data in addition to historical PV values. Several model types for processing NWPs are considered including:

- Single point downsampling
- Quadrant (4-point) geo-downsampling
- Pretrained CNN downsampling
- Region-masked downsampling

In addition to methods used for preprocessing and training a model for PV forecast, we also provide a pipeline for live model inference.


## Workflows

Runs github actions on every push

- lint.yaml: Check linting.
- test-pytest:yaml: run pytests
- build-docker.yaml: builds docker file

Runs github actions on push on main

- release-docker.yml: Builds and makes docker image release to dockerhub
