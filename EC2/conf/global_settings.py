import argparse
import os

WORKER_NUMBER = 1
FUNCTION_NAME = "new-hyperparameter-tuning"
DATA_SIZE = 5000
EPOCH = 3
PORT = 8080
OUTPUT_FILE = "log/result.txt"
BATCH_SIZE = 128
MOMENTUM = 0.9
LEARNING_RATE = 0.1

if os.environ.get("EC2_PROXY_USE_CLI") == "1":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-number", type=int, required=True)
    parser.add_argument("--function-name", type=str, required=True)
    parser.add_argument("--data-size", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--momentum", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    args = parser.parse_args()

    WORKER_NUMBER = args.worker_number
    FUNCTION_NAME = args.function_name
    DATA_SIZE = args.data_size
    EPOCH = args.epoch
    PORT = args.port
    OUTPUT_FILE = args.output_file
    BATCH_SIZE = args.batch_size
    MOMENTUM = args.momentum
    LEARNING_RATE = args.learning_rate
