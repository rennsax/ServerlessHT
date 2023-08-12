from threading import Condition, Lock
from typing import Any

from numpy import ndarray
from pydantic import BaseModel, ConfigDict, NonNegativeInt


class SyncGrad(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    receive_number: NonNegativeInt
    send_number: NonNegativeInt
    grads: list[ndarray] | None
    new_grads_hex: str | None
    cv: Condition


shared = SyncGrad(
    receive_number=0,
    send_number=0,
    grads=None,
    new_grads_hex=None,
    cv=Condition(),
)
