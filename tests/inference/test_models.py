from gradboost_pv.inference.models import NationalBoostInferenceModel, NationalPVModelConfig
from gradboost_pv.models.utils import load_nwp_coordinates


def test_national_boost_innference_model_init(model_config_path, mock_model):

    x, y = load_nwp_coordinates()
    model_config = NationalPVModelConfig.load_from_yaml(model_config_path)

    def model_loader():
        return mock_model

    _ = NationalBoostInferenceModel(nwp_x_coords=x, nwp_y_coords=y, config=model_config, model_loader=model_loader)