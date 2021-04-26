from omegaconf import OmegaConf

from padim.runner import Runner


def test_coverage() -> None:

    cfg = OmegaConf.load("./config.yaml")
    runner = Runner(cfg)
    runner.run()
