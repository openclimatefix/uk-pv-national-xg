import os

import pandas as pd
from freezegun import freeze_time
from ocf_datapipes.config.load import load_yaml_configuration

import gradboost_pv
from gradboost_pv.inference.data_feeds import ProductionDataFeed

# get main dir folder
PATH_TO_CONFIG_DIRECTORY = os.path.dirname(gradboost_pv.__file__) + "/../configs"


def test_production_datafeed(sample_prod_nwp_data, sample_prod_gsp_data, model_config_path):
    data_feed = ProductionDataFeed(path_to_configuration_file=model_config_path)
    data = {"nwp": sample_prod_nwp_data, "gsp": sample_prod_gsp_data}

    _ = data_feed.post_process(data=data)


def test_production_load_config():
    _ = load_yaml_configuration(
        filename=f"{PATH_TO_CONFIG_DIRECTORY}/default_production_datafeed.yaml"
    )



@freeze_time("2021-05-25 12:15:00")
def test_production_get_inference_time(
    sample_prod_nwp_data, sample_prod_gsp_data, model_config_path
):
    data_feed = ProductionDataFeed(path_to_configuration_file=model_config_path)

    inference_time = data_feed.get_inference_time()

    assert inference_time == pd.Timestamp("2021-05-25 11:30:00")
