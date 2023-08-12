class LambdaExit(BaseException):
    restore: bool

    def __init__(self, *args: object, restore: bool) -> None:
        super().__init__(*args)
        self.restore = restore


if __name__ == "__main__":
    ex = LambdaExit(restore=True)
    if ex.restore:
        print("restore")
