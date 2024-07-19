
from hydra.core.config_store import ConfigStore

from trainer.tasks.clip_task import CLIPTaskConfig
from trainer.tasks.mvclip_task import MVCLIPTaskConfig
from trainer.tasks.score_task import ScoreTaskConfig

cs = ConfigStore.instance()
cs.store(group="task", name="clip", node=CLIPTaskConfig)
cs.store(group="task", name="mvclip", node=MVCLIPTaskConfig)
cs.store(group="task", name="score", node=ScoreTaskConfig)
