from hydra.core.config_store import ConfigStore

from trainer.configs.configs import TrainerConfig, ScoreTrainerConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainerConfig)
cs.store(name="score_config", node=ScoreTrainerConfig)
