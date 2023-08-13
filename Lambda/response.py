from typing import Any

from pydantic import BaseModel, NonNegativeFloat


class LambdaResponse(BaseModel):
    # whether the Lambda need to be restarted
    restart: bool = False
    # whether the Lambda returns abnormally
    # alway False when restart=True
    error: bool = False
    # when error=True, record the error message
    errorMessage: str = ""
    leftTime: NonNegativeFloat = 0.0
    # the epoch that the Lambda has trained
    epoch: int = 0
    # model weight hex
    weight_hex: str | None = None

    # def __str__(self) -> str:
    #     return (
    #         f"LambdaResponse(restart={self.restart}, error={self.error}"
    #         f", errorMessage={self.errorMessage}, leftTime={self.leftTime}"
    #         f", epoch={self.epoch}"
    #         f", model_weight_hex={None if self.model_weight_hex else len(self.model_weight_hex)})"  # type: ignore
    #     )


def response_for_logging(response_dict: dict[str, Any]) -> dict[str, Any]:
    res = response_dict.copy()
    if weight_hex := res.get("model_weight_hex"):
        res.update({"model_weight_hex": f"length: {len(weight_hex)}"})
    return res
