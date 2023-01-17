import logging
from logging.config import dictConfig

logging_config = dict(
    version=1,
    formatters={
        "f": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    handlers={
        "h": {
            "class": "logging.StreamHandler",
            "formatter": "f",
            "level": logging.DEBUG,
        }
    },
    loggers={
        "": {  # root logger
            "handlers": ["h"],
            "level": logging.DEBUG,
        },
        "__main__": {  # if __name__ == '__main__'
            "handlers": ["h"],
            "level": "INFO",
            "propagate": False,
        },
    },
)


def getLogger(name: str) -> logging.Logger:
    dictConfig(logging_config)
    logger = logging.getLogger(name)
    return logger
