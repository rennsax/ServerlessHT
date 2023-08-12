import pickle

import requests
import torch.nn as nn

from exceptions import LambdaExit
from utils import get_logger, get_model_gradients, set_model_gradients

logger = get_logger(__name__)


def update_model(model: nn.Module, *, url: str = "http://127.0.0.1:8080/ps") -> None:
    """
    update the model via communicating with the parameter server

    return 0 if success, -1 if error
    """

    grads = get_model_gradients(model)
    grads_hex = pickle.dumps(grads).hex()

    # TODO hex isn't very space-efficient, consider other encoding
    logger.debug("Send request with grads size: {:d} bytes".format(len(grads_hex)))
    res = requests.post(
        # Tell the server if the Lambda need a restart
        url,
        json={"grads": grads_hex},
        timeout=None,
    )
    if res.status_code != 200:
        raise LambdaExit("While synchronizing the weight, response error occurred.")

    new_grads_hex = res.json()["new-grads"]

    logger.debug(
        "Receive response with grads size: {:d} bytes".format(len(new_grads_hex))
    )

    new_grads = pickle.loads(bytes.fromhex(new_grads_hex))
    set_model_gradients(model, new_grads)
    logger.info("Successfully update model!")
