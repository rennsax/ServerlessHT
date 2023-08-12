## Environment Variable

- `LAMBDA_LOGGING_LEVEL`: determine the logging level of the lambda function. Default: `INFO`.
- `TRAIN_LOGGING_GAP`: the mini-batch gap to log the training loss. Default: `10`.
- `LAMBDA_TOTAL_TIME` (required): the time limit of the Lambda function.
- `LAMBDA_TRAIN_LIMIT_TIME` (required): the time limit of training. Should be less than `LAMBDA_TOTAL_TIME`.