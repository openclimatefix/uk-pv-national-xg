# National XGBoost PV Model

Repository hosting various models used to predict National PV in the UK using Numerical Weather Prediction (NWP) data in addition to historical PV values. Several model types for processing NWPs are considered including:

- Single point downsampling
- Quadrant (4-point) geo-downsampling
- Pretrained CNN downsampling
- Region-masked downsampling

In addition to methods used for preprocessing and training a model for PV forecast, we also provide a pipeline for live model inference.

## NWP Data Preprocessing
Prior to any model training, we must transform the NWP zarr datasets into something useable by the model. This is done using of the downsampling methods listed above. The production model uses the region-masked downsampling.

We perform this preprocessing step prior to training rather than in an iterative DataPipes approach since XGBoost does not train well with batched data and instead prefers the full dataset at hand. In addition the downsampling reduces the size of the dataset dramatically, from a 2TB NWP zarr `xarray` dataset to a 35,000 x 34 `Pandas DataFrame`. Loading the preprocessed data from disk significantly speeds up the model training experimentation and development.

To run pre-proccessing for the region-masked model, the script `scripts/processing/uk_region_downsample.py` will take raw data from GCP and produce the processed features. To limit memory usage, the processed features are usually sliced and by year, forecast horizon step and NWP variable. Slicing by year is a common theme in all the preprocessing, since the models for each step are trained independently.

There is already preprocessed data for 2020-2021 on the GCP machine `tom-research.europe-west1-b.solar.pv-nowcasting` in `/home/tom/local_data/uk_region_nwp` which can be copied over and reused. One of the next steps for this model is to preprocess more historic data and fit the model with it - this should work with the given script above, provided the GCP filepath is changed.

## Model Training
To run model training, we need to have first pre-processed the NWP data, see above for details on that. After pre-processing the data, all downsampling methods in `gradboost_pv.models` have a common `build_dataset_from_local` method, which will parse the processed NWP features in addition to adding extra covariates - such as lagged PV values and PVlib irradiance variables.

The method `gradboost_pv.models.training.run_experiment` will fit the model to the built dataset in addition to returning some training/validation metrics. We trained on all data prior to 2021 and validated on the 2021 subset of data. By model we refer to a single model which forecasts for a single step in the future. The meta-model is an amalgamation (stored internally as a dictionary) of models, keyed by forecast horizon step (one per hour ahead). We train the models for each horizon independently.

The notebook `notebooks/models/geospatial_dsample/uk_region_model.ipynb` demonstrates the full training cycle. One can also train and upload the models using preprocessed data with the script `scripts/models/train/region_filtered_model.py`. For model finetuning, `scripts/models/grid_search_hyperparams.py` will search a parameter space of model configurations and save the results, including the best performing model.

## Accessing Current Model
For access to the current models, they are stored in the s3 bucket: `nowcasting-national-forecaster-models-development` and models are keyed by the forecast hour they refer to. They can also be access via `gradboost_pv.models.s3.load_model`

## Model Inference
There are two ways to run model inference. A draft version, which builds a mock datapipe from GCP data can be found at `scripts/inference/mock_setup.py` and can be ran using 
```
python -m scripts.inference.mock_setup
```

For inference using production datafeeds, running the following will give model prediction: 
```
python -m gradboost_pv.app --path_to_model_config <PATH_TO_MODEL_CONFIG> --path_to_datafeed_config <PATH_TO_DATAFEED_CONFIG>
```
There are default configuation files for production inference in `configs`.

## Model Specification
### Inputs
The model is trained with NWP and GSP National PV data from GCP. There are some differences between this data and that of the production datafeeds that need to be addressed. The current model is trained using preprocessed channels `['dswrf', 'hcct', 'lcc', 't', 'sde', 'wdir10']`. The following specification seems to be the most up to date description of the training dataset parameters:
```
        channels: The NWP forecast parameters to load. If None then don't filter.
            See:  http://cedadocs.ceda.ac.uk/1334/1/uk_model_data_sheet_lores1.pdf
            All of these params are "instant" (i.e. a snapshot at the target time,
            not accumulated over some time period).  The available params are:
                cdcb  : Height of lowest cloud base > 3 oktas, in meters above surface.
                lcc   : Low-level cloud cover in %.
                mcc   : Medium-level cloud cover in %.
                hcc   : High-level cloud cover in %.
                sde   : Snow depth in meters.
                hcct  : Height of convective cloud top, meters above surface.
                        WARNING: hcct has NaNs where there are no clouds forecast to exist!
                dswrf : Downward short-wave radiation flux in W/m^2 (irradiance) at surface.
                dlwrf : Downward long-wave radiation flux in W/m^2 (irradiance) at surface.
                h     : Geometrical height, meters.
                t     : Air temperature at 1 meter above surface in Kelvin.
                r     : Relative humidty in %.
                dpt   : Dew point temperature in Kelvin.
                vis   : Visibility in meters.
                si10  : Wind speed in meters per second, 10 meters above surface.
                wdir10: Wind direction in degrees, 10 meters above surface.
                prmsl : Pressure reduce to mean sea level in Pascals.
                prate : Precipitation rate at the surface in kg/m^2/s.
```

WARNING: Currently the model is running in production with only the variables: `['dswrf', 'lcc', 't']`, this will likely result in model underperformance.

The default nature of the XGBoost is that missing features of the dataset do not necessarily break the model, due it its tree based nature, `nan` values can pass through ignored. However, the model was never trained with missing data/features and so the model never learnt to adapt to this. Thus, key components of the tree structure might be heavily sensitive to missing features with respect to predictive performance.

## Configurations
Model inference can be configured using a `yaml` file or `dataclass` object `NationalPVModelConfig`. In this configuration, we can specify which forecast horizons we should be expecting (a subset of `range(36)`), what NWP variables we should be selecting (via `nwp_variables`) and whether we should allow for missing features from our expected/full training features list (`allow_missing_covariates` and `required_model_covariates` respectively).

The model used for prediction is `XGBRegressor` from `XGBoost` library (https://xgboost.readthedocs.io/en/stable/), with the following hyperparameter configuration:
```
hyperparams = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "colsample_bylevel": 1,
    "colsample_bynode": 1,
    "colsample_bytree": 0.85,
    "early_stopping_rounds": None,
    "gamma": 0,
    "gpu_id": -1,
    "grow_policy": "depthwise",
    "importance_type": None,
    "interaction_constraints": "",
    "learning_rate": 0.01,
    "max_bin": 256,
    "max_cat_threshold": 64,
    "max_depth": 80,
    "max_leaves": 0,
    "min_child_weight": 5,
    "n_estimators": 750,
    "n_jobs": -1,
    "num_parallel_tree": 1,
    "predictor": "auto",
    "random_state": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "sampling_method": "uniform",
    "scale_pos_weight": 1,
    "subsample": 0.65,
    "tree_method": "hist",
    "validate_parameters": 1,
    "verbosity": 1,
}
```


## Questions

- How to run training? What scripts to run
- Description of the current model
  - What data does it take in? What NWP channels does it use
  - What is the pre processing
  - What ml model is used?
- What happens if there is missing data?
- Where is the model saved?
- How to run inferences? 

## Workflows

Runs github actions on every push

- lint.yaml: Check linting.
- test-pytest:yaml: run pytests
- build-docker.yaml: builds docker file

Runs github actions on push on main

- release-docker.yml: Builds and makes docker image release to dockerhub
