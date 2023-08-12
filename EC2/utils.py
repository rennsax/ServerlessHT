import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = True  # default to be True in fact
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(
        logging.Formatter(
            "{asctime} | {name} | L{lineno} | {levelname}\n {message}\n",
            datefmt="%Y-%m-%d %H:%M:%S",
            style="{",
        )
    )
    logger.addHandler(stream_handler)
    return logger
