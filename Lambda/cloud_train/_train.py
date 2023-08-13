import logging
import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision  # type: ignore
from torch.utils.data import DataLoader, Subset

from exceptions import LambdaExit
from hyperparameter import Hyperparameter
from utils import get_logger, predict_if_restart

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


def get_data_loader(batch_size: int, slice_range: tuple[int, int]):
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
    test_set = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


def train_model(
    model: nn.Module,
    hyperparameter: Hyperparameter,
    *,
    begin_epoch: int,
    total_epoch: int,
    slice_range: tuple[int, int],
    proxy_url: str,
    get_remaining_time: Callable[[], int],
):
    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparameter.learning_rate,
        momentum=hyperparameter.momentum,
    )

    train_loader, test_loader = get_data_loader(hyperparameter.batch_size, slice_range)

    model.train()
    logging_gap: int = int(os.environ.get("TRAIN_LOGGING_GAP", 10))

    for epoch in range(begin_epoch, total_epoch):
        for i, (train_x, train_label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(train_x)
            loss = loss_function(output, train_label)
            loss.backward()
            optimizer.step()
            if i % logging_gap == 0:
                _train_logger.info(
                    f"Epoch {epoch + 1}, step {i}, loss: {loss.item():.3f}"
                )

        # Each epoch, sync the weight with parameter server
        _logger.info("Epoch %d, sync weight with parameter server", epoch)
        update_model(model, url=proxy_url)

        if epoch != total_epoch - 1 and predict_if_restart(epoch, get_remaining_time()):
            raise LambdaExit(restore=True, cur_epoch=epoch)

    model.eval()
    # test the model
    correct = 0
    total = 0
    for test_x, test_label in test_loader:
        output = model(test_x)
        _, predicted = torch.max(output.data, 1)
        total += test_label.size(0)
        correct += (predicted == test_label).sum().item()

    test_accuracy = correct / total
    _train_logger.info("Test accuracy: %.2f %%", 100 * test_accuracy)
    return test_accuracy
