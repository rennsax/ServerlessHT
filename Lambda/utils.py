import logging
import os

import torch
import torch.nn as nn

utils_logger = logging.getLogger(__name__)


def get_model_weight(model: nn.Module):
    return {k: v.cpu() for k, v in model.state_dict().items()}


def set_model_weight(model: nn.Module, weights):
    model.load_state_dict(weights)


def get_model_gradients(model: nn.Module):
    grads = []
    for p in model.parameters():
        grad = None if p.grad is None else p.grad.data.cpu().numpy()
        grads.append(grad)
    return grads


def set_model_gradients(model, gradients):
    for g, p in zip(gradients, model.parameters()):
        if g is not None:
            p.grad = torch.from_numpy(g)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    class CustomFormatter(logging.Formatter):
        grey = "\x1b[30;20m"
        white = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        format_str = "[{asctime} | {name} | L{lineno} | {levelname}] {message}"

        FORMATS = {
            logging.DEBUG: grey + format_str + reset,
            logging.INFO: white + format_str + reset,
            logging.WARNING: yellow + format_str + reset,
            logging.ERROR: red + format_str + reset,
            logging.CRITICAL: bold_red + format_str + reset,
        }

        def format(self, record: logging.LogRecord):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(
                log_fmt,
                datefmt="%Y-%m-%d %H:%M:%S",
                style="{",
            )

            return formatter.format(record)

    stream_handler.setFormatter(CustomFormatter())
    logger.addHandler(stream_handler)
    return logger


predict_logger = get_logger("utils.predict_restart")


def predict_if_restart(cur_epoch: int, remaining_time: int) -> bool:
    total_time = int(os.environ.get("LAMBDA_TOTAL_TIME")) * 1000  # type: ignore
    train_time_limit = int(os.environ.get("LAMBDA_TRAIN_LIMIT_TIME")) * 1000  # type: ignore
    assert train_time_limit <= total_time

    remaining_time -= total_time - train_time_limit
    spent_percentage = 1 - remaining_time / train_time_limit
    average_per_epoch = spent_percentage / (cur_epoch + 1)
    predict_logger.debug(
        "Predict result: epoch %d, %.4f -> %.4f",
        cur_epoch,
        spent_percentage,
        spent_percentage + average_per_epoch,
    )
    if spent_percentage + average_per_epoch >= 1:
        return True
    return False
