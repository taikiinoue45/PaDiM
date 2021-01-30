from padim.runner import BaseRunner


class Runner(BaseRunner):
    def _train(self, epoch: int) -> None:
        pass

    def _validate(self, epoch: int) -> float:
        pass

    def _test(self, epoch: int) -> None:
        pass
