import asyncio
import json
from typing import Any

import numpy as np

from constants import design_space
from models import Hyperparameter
from train import train

with open("config.json", "r") as f:
    config: dict[str, Any] = json.load(f)

MAX_EVALUATED_INDIVIDUAL: int = config["genetic.maxEvaluatedIndividual"]

def random_hyperparameter() -> Hyperparameter:
    np.random.choice(design_space["batch_size"])
    return Hyperparameter(
        **{param_name: np.random.choice(space) for param_name, space in design_space.items() }
    )

async def main() -> float:
    total_time = 0
    for i in range(MAX_EVALUATED_INDIVIDUAL):
        params = random_hyperparameter()
        res = await train(params, i)
        print(params, res)
        total_time += res[1]
        if res[0] > 0.95:
            break
    return total_time



if __name__ == "__main__":
    print("Total time:", asyncio.run(main()))