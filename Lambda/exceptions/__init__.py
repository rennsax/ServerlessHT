class LambdaExit(BaseException):
    restore: bool
    cur_epoch: int

    def __init__(
        self, *args: object, restore: bool = False, cur_epoch: int = 0
    ) -> None:
        super().__init__(*args)
        self.restore = restore
        self.cur_epoch = cur_epoch
