from omegaconf import OmegaConf
from py._path.local import LocalPath

from padim.runner import Runner


def test_run(tmpdir: LocalPath) -> None:

    cfg = OmegaConf.load("./config.yaml")
    runner = Runner(cfg)
    runner.run()
