import argparse
import logging
import os
import pickle
import time
from typing import Any

import exceptions
from cloud_train import train_model
from hyperparameter import Hyperparameter
from model import LambdaModel
from response import LambdaResponse, response_for_logging
from utils import get_logger, get_model_weight, set_model_weight


class AWSLambdaContext:
    def get_remaining_time_in_millis(self) -> int:  # type: ignore
        ...

    ...


app_logger = get_logger(__name__)


def check_required_env(*evs: str) -> None:
    for ev in evs:
        if os.environ.get(ev) is None:
            app_logger.error("Unset environment variable: %s", ev)
            raise exceptions.LambdaExit("Unset environment variable: %s" % ev)


def handler(event, context: AWSLambdaContext) -> dict[str, Any]:
    response = LambdaResponse()
    try:
        check_required_env("LAMBDA_TOTAL_TIME", "LAMBDA_TRAIN_LIMIT_TIME")

        # set all loggers according to the environment variable
        logging_level = os.environ.get("LAMBDA_LOGGING_LEVEL", "INFO")
        # fragile next line
        logging_level_id = getattr(logging, logging_level, logging.INFO)
        loggers = [
            app_logger,
            logging.getLogger("cloud_train.sync_weight"),
            logging.getLogger("cloud_train._train"),
            logging.getLogger("cloud_train._train.train_info"),
            logging.getLogger("utils"),
        ]
        for logger in loggers:
            logger.setLevel(logging_level_id)

        app_logger.info("Use logging level: %s", logging_level)

        total_time: int = int(os.environ.get("LAMBDA_TOTAL_TIME"))  # type: ignore
        train_time_limit: int = int(os.environ.get("LAMBDA_TRAIN_LIMIT_TIME"))  # type: ignore
        if train_time_limit > total_time:
            app_logger.warning(
                "The environment variable LAMBDA_TOTAL_TIME"
                "is less than LAMBDA_TRAIN_LIMIT_TIME"
                ", set train time limit to lambda total time limit"
            )
            os.environ["LAMBDA_TRAIN_LIMIT_TIME"] = str(total_time)

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
            begin_epoch = int(event["begin-epoch"])
        except KeyError as e:
            raise exceptions.LambdaExit(
                "Lambda handler receives an event without expected key: {}".format(e),
            ) from e
        except ValueError as e:
            raise exceptions.LambdaExit(
                "Lambda handler receives an event with invalid value"
            ) from e

        hyperparameter = Hyperparameter(
            batch_size=batch_size,
            learning_rate=lr,
            momentum=momentum,
        )
        model = LambdaModel()

        if weight_hex := event.get("model_weight_hex") is not None:
            app_logger.debug("Loading model weight...")
            weight = pickle.loads(bytes.fromhex(weight_hex))  # type: ignore
            set_model_weight(model, weight)

        try:
            train_model(
                model,
                hyperparameter,
                total_epoch=epoch,
                begin_epoch=begin_epoch,
                slice_range=(slice_begin, slice_end),
                proxy_url=proxy_url,
                get_remaining_time=context.get_remaining_time_in_millis,
            )
        except exceptions.LambdaExit as ex:
            if ex.restore:
                # Lambda need to be restarted
                response.restart = True
                response.epoch = ex.cur_epoch + 1
                model_weight = get_model_weight(model)
                model_weight_hex = pickle.dumps(model_weight).hex()
                response.weight_hex = model_weight_hex
                app_logger.debug("Model weight size: %d bytes", len(model_weight_hex))
            else:
                raise ex
    except exceptions.LambdaExit as ex:
        assert not ex.restore
        app_logger.exception(ex)
        response.error = True
        response.errorMessage = str(ex)
    except BaseException as ex:
        app_logger.error("An unexpected error %s occurs", ex)
        response.error = True
        response.errorMessage = str(ex)

    remaining_time_in_seconds = context.get_remaining_time_in_millis() / 1000
    app_logger.info("Remaining time: %.2f", remaining_time_in_seconds)
    response.leftTime = remaining_time_in_seconds

    app_logger.debug("Lambda response: %s", response_for_logging(response))

    return response.model_dump()


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
        "begin-epoch": 0,
    }

    LAMBDA_TIME = 10  # Lambda time limit in seconds

    class FakeContext:
        def __init__(self, end_time) -> None:
            self.function_name = "hello, Lambda!"
            self._end_time = end_time

        def get_remaining_time_in_millis(self) -> int:
            return (self._end_time - time.time()) * 1000

    fake_context = FakeContext(time.time() + LAMBDA_TIME)

    res = handler(sample_event, fake_context)  # type: ignore

    print("Lambda response: ", res)
