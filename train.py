import asyncio
import hashlib
import json
import logging
import pathlib
from typing import Any

from models import Hyperparameter

logger = logging.getLogger(__name__)
logger.propagate = True  # default to be True in fact
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(
    logging.Formatter(
        "{asctime} | {name} | {levelname}\n {message}\n",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )
)
logger.addHandler(stream_handler)


def hash_hyperparameter(params: Hyperparameter) -> str:
    m = hashlib.sha1()
    m.update(str(params).encode())
    return m.hexdigest()


async def train(params: Hyperparameter, index: int) -> tuple[float, ...]:
    output_file = pathlib.Path("output/result_" + str(index) + ".txt")
    log_output = pathlib.Path("subprocess/" + str(index) + ".txt")
    command: list[str] = [
        "zsh",
        "launch_faastuning.zsh",
        "4",  # worker number
        "new-hyperparameter-tuning",  # function name
        "60000",  # data size
        "20",  # epoch
        f"{10000+index}",  # port
        str(index),
        str(params.batch_size),
        str(params.momentum),
        str(params.learning_rate),
    ]
    log_file = open(log_output, "w")
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=log_file,
        stderr=log_file,
    )
    logger.info("Create process with command: %s", " ".join(command))
    await proc.wait()
    logger.info("Process %d finished", index)
    log_file.close()
    try:
        with open(output_file, "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            raise RuntimeError("no output")
        res = lines[0].strip().split(" ")
        if len(res) != 2:
            raise RuntimeError("invalid output")
        logger.info("Train (%s) over with result %s", params, res)
    except Exception:
        logger.exception("Train (%s) failed", params)
        return (0.0, 0.0, 0.0)
    else:
        return (*map(float, res), 0)


async def main():
    logger.info(
        await train(
            Hyperparameter(
                batch_size=128,
                learning_rate=0.001,
                momentum=0.9,
            ),
            0,
        )
    )


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    res = asyncio.run(main())
