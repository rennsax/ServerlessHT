# from server import app
import logging

from initialize import create_worker

if __name__ == "__main__":
    # app.run(host="0.0.0.0", debug=True, port=8080)
    logging.getLogger("initialize").setLevel(logging.DEBUG)
    thread_list = create_worker()
    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()
