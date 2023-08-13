from . import global_settings


class Settings:
    WORKER_NUMBER: int
    FUNCTION_NAME: str
    DATA_SIZE: int
    EPOCH: int
    PORT: int
    OUTPUT_FILE: str

    def __init__(self, settings):
        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))


settings = Settings(global_settings)
