import argparse
import logging
import os
import time

import exceptions
from cloud_train import train_model
from hyperparameter import Hyperparameter
from model import LambdaModel
from utils import get_logger

app_logger = get_logger(__name__)


class AWSLambdaContext:
    def get_remaining_time_in_millis(self) -> int:  # type: ignore
        ...

    ...


def handler(event, context: AWSLambdaContext):
    # set all loggers according to the environment variable
    logging_level = os.environ.get("LAMBDA_LOGGING_LEVEL", "INFO")
    # fragile next line
    logging_level_id = getattr(logging, logging_level, logging.INFO)
    loggers = [
        app_logger,
        logging.getLogger("cloud_train.sync_weight"),
        logging.getLogger("cloud_train._train"),
        logging.getLogger("cloud_train._train.train_info"),
    ]
    for logger in loggers:
        logger.setLevel(logging_level_id)
    app_logger.info("Use logging level: %s", logging_level)

    # parse event
    app_logger.debug("Receives event: %s", event)
    try:
        lr = float(event["learning-rate"])
        batch_size = int(event["batch-size"])
        momentum = float(event["momentum"])
        slice_begin = int(event["slice-begin"])
        slice_end = int(event["slice-end"])
        epoch = int(event["epoch"])
        proxy_url = event["proxy-url"]
    except KeyError as e:
        raise RuntimeError(
            "Lambda handler receives an event without expected key: {}".format(e),
        ) from e
    except ValueError as e:
        raise RuntimeError("Lambda handler receives an event with invalid value") from e

    hyperparameter = Hyperparameter(
        batch_size=batch_size,
        learning_rate=lr,
        momentum=momentum,
    )
    model = LambdaModel()

    try:
        train_model(
            model,
            hyperparameter,
            total_epoch=epoch,
            slice_range=(slice_begin, slice_end),
            proxy_url=proxy_url,
        )
    except exceptions.LambdaExit as ex:
        # TODO
        if ex.restore:
            app_logger.info("Restore model")
        else:
            app_logger.exception(ex)
    except BaseException as ex:
        app_logger.error("An unexpected error %s occurs", ex)

    app_logger.info(
        "Remaining time: %.2f", context.get_remaining_time_in_millis() / 1000
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker",
        type=str,
        required=True,
        help="format: [current]/[total]",
    )
    parser.add_argument(
        "--data-size",
        type=int,
        default=60000,
        required=False,
        help="the length of the train data",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=2,
        required=False,
    )
    args = parser.parse_args()
    idx, total = map(int, args.worker.split("/"))
    slice_begin = idx * args.data_size // total
    slice_end = (idx + 1) * args.data_size // total

    sample_event = {
        "proxy-url": "http://127.0.0.1:8080/ps",
        "slice-begin": str(slice_begin),
        "slice-end": str(slice_end),
        "epoch": str(args.epoch),
        "learning-rate": "0.01",
        "batch-size": "128",
        "momentum": "0.9",
    }

    LAMBDA_TIME = 900  # Lambda time limit in seconds

    class FakeContext:
        def __init__(self, end_time) -> None:
            self.function_name = "hello, Lambda!"
            self._end_time = end_time

        def get_remaining_time_in_millis(self) -> int:
            return (self._end_time - time.time()) * 1000

    fake_context = FakeContext(time.time() + LAMBDA_TIME)

    handler(sample_event, fake_context)  # type: ignore
