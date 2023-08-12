# from server import app
import logging
import os
import signal

from conf import settings
from initialize import create_worker
from utils import get_logger

logger = get_logger(__name__)


def siguser1_handler(signal, frame):
    logger.info("The process is killed by SIGUSR1 (a Lambda acted abnormally)")
    os._exit(0)


if __name__ == "__main__":
    # app.run(host="0.0.0.0", debug=True, port=8080)
    signal.signal(signal.SIGUSR1, siguser1_handler)

    logging.getLogger("initialize").setLevel(logging.DEBUG)
    thread_list = create_worker(settings.WORKER_NUMBER)
    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()
