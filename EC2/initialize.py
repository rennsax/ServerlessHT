import json
import threading
from typing import Any

import boto3
import requests
import utils
from conf import settings

logger = utils.get_logger(__name__)
lambda_client = boto3.client("lambda")


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


def invoke_lambda(index: int, payload: dict[str, Any]):
    # validate the Lambda function exists

    response = lambda_client.invoke(
        FunctionName=settings.FUNCTION_NAME,
        Payload=json.dumps(payload).encode("utf-8"),
        # InvocationType="DryRun",
    )
    logger.info("The %d-th worker response with %s", index, response)


def create_worker() -> list[threading.Thread]:
    validate_invoke()

    # @see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
    instance_ip = requests.get(
        "http://169.254.169.254/latest/meta-data/public-ipv4"
    ).text

    if instance_ip == "":
        # not on EC2, assume a local test
        instance_ip = "127.0.0.1"

    thread_list: list[threading.Thread] = list()
    for i in range(settings.WORKER_NUMBER):
        slice_begin = i * settings.DATA_SIZE // settings.WORKER_NUMBER
        slice_end = (i + 1) * settings.DATA_SIZE // settings.WORKER_NUMBER
        payload = {
            "proxy-url": f"http://{instance_ip}:{settings.PORT}/ps",
            "slice-begin": slice_begin,
            "slice-end": slice_end,
            "epoch": settings.EPOCH,
            "learning-rate": 0.01,
            "batch-size": 128,
            "momentum": 0.9,
        }
        thread_list.append(threading.Thread(target=invoke_lambda, args=(i, payload)))

    return thread_list


if __name__ == "__main__":
    for thread in create_worker():
        thread.join()
