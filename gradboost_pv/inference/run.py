from typing import Iterator, Callable, Dict

import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from gradboost_pv.inference.models import ModelOutput


@functional_datapipe("nationalboost_model_inference")
class NationalBoostModelInference(IterDataPipe):
    def __init__(
        self,
        model_loader: Callable[[], Callable[[pd.DataFrame], ModelOutput]],
        data_feed: IterDataPipe,
        data_processor: Callable[[Dict[str, xr.Dataset]], pd.DataFrame],
    ) -> None:
        self.model_loader = model_loader
        self.model = self.model_loader()
        self.data_feed = data_feed
        self.data_processor = data_processor

    def __iter__(self) -> Iterator[ModelOutput]:
        for data in self.data_feed:
            processed = self.data_processor(data)
            if processed is not None:  # model may need a warm up period to start
                output: ModelOutput = self.model(processed)
                yield output
