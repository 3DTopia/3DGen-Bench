from hydra.core.config_store import ConfigStore

from trainer.models.clip_model import ClipModelConfig
from trainer.models.mvclip_model import MVCLIPModelConfig
from trainer.models.score_model import ScoreModelConfig

cs = ConfigStore.instance()
cs.store(group="model", name="clip", node=ClipModelConfig)
cs.store(group="model", name="mvclip", node=MVCLIPModelConfig)
cs.store(group="model", name="score", node=ScoreModelConfig)
