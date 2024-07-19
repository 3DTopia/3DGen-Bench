from hydra.core.config_store import ConfigStore

from trainer.datasetss.dataset import MVDatasetConfig
from trainer.datasetss.score_dataset import ScoreDatasetConfig

cs = ConfigStore.instance()
cs.store(group="dataset", name="mvclip", node=MVDatasetConfig)
cs.store(group="dataset", name="score", node=ScoreDatasetConfig)
