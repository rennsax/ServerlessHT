import logging
import pickle
import threading

import numpy as np
from flask import Flask, request

from conf import settings
from global_v import shared

app = Flask(__name__)

WORKER_NUMBER = settings.WORKER_NUMBER

# suppress WSGI logging
flask_wsgi_logger = logging.getLogger("werkzeug")
flask_wsgi_logger.setLevel(logging.ERROR)


def after_all_response():
    with shared.cv:
        shared.send_number += 1
        if shared.send_number != WORKER_NUMBER:
            shared.cv.wait()
        else:
            app.logger.debug("All responses sent, now clean the data")
            shared.grads, shared.new_grads_hex = None, None
            shared.receive_number, shared.send_number = 0, 0
            shared.cv.notify_all()


@app.post("/ps")
def sync_grads():
    grads_hex: str = request.json["grads"]  # type: ignore
    app.logger.debug("Receive request, grads: {:d} bytes".format(len(grads_hex)))
    grads: list[np.ndarray] = pickle.loads(bytes.fromhex(grads_hex))  # type: ignore
    with shared.cv:
        if shared.grads is None:
            shared.grads = grads
        else:
            for i in range(len(grads)):
                shared.grads[i] += grads[i]
        shared.receive_number += 1
        if shared.receive_number != WORKER_NUMBER:
            shared.cv.wait_for(lambda: shared.receive_number == WORKER_NUMBER)
        else:
            for grad in shared.grads:
                grad /= WORKER_NUMBER
            assert shared.new_grads_hex is None, "new_grads_hex should be None"
            shared.new_grads_hex = pickle.dumps(shared.grads).hex()
            shared.cv.notify_all()

    app.logger.debug(
        "Return grads with size {:d} bytes".format(len(shared.new_grads_hex))
    )
    # clear
    new_grads_hex = shared.new_grads_hex
    thr = threading.Thread(target=after_all_response)
    thr.start()
    return {
        "new-grads": new_grads_hex,
    }


@app.get("/check")
def check_variable():
    app.logger.debug("Checked: {}".format(shared))
    return {"syncGrad": shared.__repr__()}
