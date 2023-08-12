from . import global_settings


class Settings:
    WORKER_NUMBER: int
    FUNCTION_NAME: str
    DATA_SIZE: int
    EPOCH: int
    PORT: int

    def __init__(self, settings):
        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))


settings = Settings(global_settings)
