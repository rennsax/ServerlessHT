import argparse
import os

WORKER_NUMBER = 1
FUNCTION_NAME = "new-hyperparameter-tuning"
DATA_SIZE = 60000
EPOCH = 1
PORT = 8080

if os.environ.get("EC2_PROXY_USE_CLI") == "1":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-number", type=int, required=True)
    parser.add_argument("--function-name", type=str, required=True)
    parser.add_argument("--data-size", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    WORKER_NUMBER = args.worker_number
    FUNCTION_NAME = args.function_name
    DATA_SIZE = args.data_size
    EPOCH = args.epoch
    PORT = args.port
