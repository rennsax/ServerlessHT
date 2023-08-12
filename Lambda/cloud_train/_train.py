import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision  # type: ignore
from hyperparameter import Hyperparameter
from torch.utils.data import DataLoader, Subset
from utils import get_logger

from .sync_weight import update_model

torch.random.manual_seed(0)
np.random.seed(0)

_logger = get_logger(__name__)

_train_logger = _logger.getChild("train_info")
_train_logger.propagate = False
_train_logger_handler = logging.StreamHandler()
_train_logger_handler.setLevel(logging.DEBUG)
_train_logger_handler.setFormatter(
    logging.Formatter(
        "[{asctime} | training info]: {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
_train_logger.addHandler(_train_logger_handler)


def get_train_data_loader(batch_size: int, slice_range: tuple[int, int]):
    train_set = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )
    train_set_slice = Subset(train_set, np.arange(*slice_range))
    train_loader = DataLoader(
        train_set_slice,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    _logger.info("Worker get data slice %s", slice_range)

    return train_loader


def train_model(
    model: nn.Module,
    hyperparameter: Hyperparameter,
    total_epoch: int,
    slice_range: tuple[int, int],
    proxy_url: str,
):
    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparameter.learning_rate,
        momentum=hyperparameter.momentum,
    )

    train_loader = get_train_data_loader(hyperparameter.batch_size, slice_range)

    model.train()
    logging_gap: int = int(os.environ.get("TRAIN_LOGGING_GAP", 10))

    for epoch in range(1, total_epoch + 1):
        for i, (train_x, train_label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(train_x)
            loss = loss_function(output, train_label)
            loss.backward()
            optimizer.step()
            if i % logging_gap == 0:
                _train_logger.info(f"Epoch {epoch}, step {i}, loss: {loss.item():.3f}")

        # Each epoch, sync the weight with parameter server
        update_model(model, url=proxy_url)
