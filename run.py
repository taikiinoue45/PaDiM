import logging
import os
import sys
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig

from padim.runner import Runner


log = logging.getLogger(__name__)

config_path = str(Path(sys.argv[1]).parent)
config_name = str(Path(sys.argv[1]).stem)
sys.argv.pop(1)


@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:

    mlflow.set_tracking_uri(cfg.params.tracking_uri)
    mlflow.set_experiment(cfg.params.experiment_name)
    mlflow.start_run(run_name=cfg.params.run_name)
    mlflow.log_params(cfg.params)
    mlflow.log_param("cwd", os.getcwd())
    mlflow.log_artifacts(".hydra", "hydra")

    runner = Runner(cfg)
    runner.run()


if __name__ == "__main__":

    run()
