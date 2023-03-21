from gradboost_pv.inference.data_feeds import ProductionDataFeed


def test_production_datafeed(sample_prod_nwp_data, sample_prod_gsp_data, model_config_path):
    data_feed = ProductionDataFeed(path_to_configuration_file=model_config_path)
    data = {"nwp": sample_prod_nwp_data, "gsp": sample_prod_gsp_data}

    _ = data_feed.post_process(data=data)
