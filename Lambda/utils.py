import logging

import torch
import torch.nn as nn


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
    stream_handler.setFormatter(
        logging.Formatter(
            # To better navigate the logging in CloudWatch, merge into one line
            "[{asctime} | {name} | L{lineno} | {levelname}] {message}",
            datefmt="%Y-%m-%d %H:%M:%S",
            style="{",
        )
    )
    logger.addHandler(stream_handler)
    return logger
