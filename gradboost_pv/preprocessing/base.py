from abc import ABC, abstractmethod

from gradboost_pv.utils.typing import Hour, ProcessedNWP


class BasePreProcessedNWPLoader(ABC):
    @abstractmethod
    def __call__(self, forecast_horizon_hour: Hour) -> ProcessedNWP:
        pass
