from hydra.core.config_store import ConfigStore

from trainer.criterions.criterion import MVCriterionConfig
from trainer.criterions.score_criterion import ScoreCriterionConfig


cs = ConfigStore.instance()
cs.store(group="criterion", name="mvclip", node=MVCriterionConfig)
cs.store(group="criterion", name="score", node=ScoreCriterionConfig)

