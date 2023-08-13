import json
import os
import signal
import threading
from threading import Lock
from typing import Any

import boto3
import requests
import utils
from conf import settings
from response import LambdaResponse, response_for_logging

logger = utils.get_logger(__name__)
lambda_client = boto3.client("lambda")
test_results = []
lambda_total_time: list[float] = []
lambda_total_time_mtx = Lock()

TOTAL_TIME_LIMIT = 900


def validate_invoke() -> bool:
    # validate the Lambda function exists
    try:
        lambda_client.get_function(FunctionName=settings.FUNCTION_NAME)

        sample_payload = {
            "proxy-url": "http://127.0.0.1:10000/ps",
            "slice-begin": 0,
            "slice-end": 128,
            "epoch": 1,
            "learning-rate": "0.01",
            "batch-size": "128",
            "momentum": "0.9",
            "begin-epoch": 0,
        }

        lambda_client.invoke(
            FunctionName=settings.FUNCTION_NAME,
            Payload=json.dumps(sample_payload).encode("utf-8"),
            InvocationType="DryRun",
        )
    except Exception as ex:
        logger.exception(ex)
        return False

    return True


def add_time(_time: float):
    with lambda_total_time_mtx:
        lambda_total_time.append(_time)


def invoke_lambda(index: int, payload: dict[str, Any]):
    # validate the Lambda function exists

    test_res = "0"
    while True:
        logger.debug(
            "Invoke %d-th worker with %s", index, response_for_logging(payload)
        )
        boto3_response = lambda_client.invoke(
            FunctionName=settings.FUNCTION_NAME,
            Payload=json.dumps(payload).encode("utf-8"),
        )
        response = json.loads(boto3_response["Payload"].read())
        logger.debug(
            "The %d-th worker response with %s", index, response_for_logging(response)
        )
        add_time(TOTAL_TIME_LIMIT - float(response["leftTime"]))

        if response["error"]:
            logger.error(
                "The %d-th worker fails: %s",
                index,
                response["errorMessage"],
            )
            # os.kill(os.getpid(), signal.SIGUSR1)
            break
        if not response["restart"]:
            logger.info("The %d-th worker finishes", index)
            test_res = response.get("test_accuracy", "0")
            break
        payload["begin-epoch"] = response["epoch"]
        if (weight_hex := response.get("weight_hex")) is not None:
            payload["weight_hex"] = weight_hex
        logger.info("The %d-th worker restarts from epoch %d", index, response["epoch"])
    test_results.append(test_res)


def create_worker(worker_number: int) -> list[threading.Thread]:
    validate_invoke()

    # @see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
    instance_ip = requests.get(
        "http://169.254.169.254/latest/meta-data/public-ipv4"
    ).text

    if instance_ip == "":
        # not on EC2, assume a local test
        instance_ip = "127.0.0.1"

    thread_list: list[threading.Thread] = list()
    for i in range(worker_number):
        slice_begin = i * settings.DATA_SIZE // worker_number
        slice_end = (i + 1) * settings.DATA_SIZE // worker_number
        payload = {
            "proxy-url": f"http://{instance_ip}:{settings.PORT}/ps",
            "slice-begin": slice_begin,
            "slice-end": slice_end,
            "epoch": settings.EPOCH,
            "learning-rate": settings.LEARNING_RATE,
            "batch-size": settings.BATCH_SIZE,
            "momentum": settings.MOMENTUM,
            # 0-indexed
            "begin-epoch": 0,
        }
        thread_list.append(threading.Thread(target=invoke_lambda, args=(i, payload)))

    return thread_list


if __name__ == "__main__":
    for thread in create_worker(settings.WORKER_NUMBER):
        thread.join()
