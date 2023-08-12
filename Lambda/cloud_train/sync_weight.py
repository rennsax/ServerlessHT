import logging
import pickle

import requests
import torch.nn as nn
from exceptions import LambdaExit
from utils import get_logger, get_model_gradients, set_model_gradients

logger = get_logger(__name__)


def update_model(model: nn.Module, *, url: str = "http://127.0.0.1:8080/ps") -> None:
    grads = get_model_gradients(model)
    grads_hex = pickle.dumps(grads).hex()

    # TODO hex isn't very space-efficient, consider other encoding
    logger.debug("Send request with grads size: {:d} bytes".format(len(grads_hex)))
    res = requests.post(url, json={"grads": grads_hex}, timeout=None)
    if res.status_code != 200:
        logging.error("Response error!")
        raise LambdaExit("An unexpected error occurs", restore=False)

    new_grads_hex = res.json()["new-grads"]
    is_restart = res.json().get("restart", False)
    if is_restart:
        logger.info("Restarted by the server response")
        raise LambdaExit(restore=True)
    logger.debug(
        "Receive response with grads size: {:d} bytes".format(len(new_grads_hex))
    )
    new_grads = pickle.loads(bytes.fromhex(new_grads_hex))
    set_model_gradients(model, new_grads)
    logger.info("Successfully update model!")
