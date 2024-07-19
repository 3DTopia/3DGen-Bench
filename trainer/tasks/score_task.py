import collections
from dataclasses import dataclass
from typing import List

import torch
import numpy as np
from PIL import Image
from accelerate.logging import get_logger
from accelerate.utils import LoggerType
from omegaconf import II
from transformers import AutoTokenizer

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.tasks.base_task import BaseTaskConfig, BaseTask

logger = get_logger(__name__)

EVAL_DIMS = ["geometry", "geo_detail", "texture", "geo_texture", "alignment"]

@dataclass
class ScoreTaskConfig(BaseTaskConfig):
    _target_: str = "trainer.tasks.score_task.ScoreTask"
    pretrained_clip_model_name_or_path: str = II("model.mv_clip.pretrained_clip_model_name_or_path")
    label_column_name: str = II("dataset.label_column_name")

    reference_type_column_name: str = II("dataset.reference_type_column_name")
    reference_input_column_name: str = II("dataset.reference_input_column_name")
    reference_idx_column_name: str = II("dataset.reference_idx_column_name")
    normal_pixels_column_name: str = II("dataset.normal_pixels_column_name")
    rgb_pixels_column_name: str = II("dataset.rgb_pixels_column_name")

def numpy_to_pil(images):
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


class ScoreTask(BaseTask):
    def __init__(self, cfg: ScoreTaskConfig, accelerator: BaseAccelerator):
        super().__init__(cfg, accelerator)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_clip_model_name_or_path)
        self.cfg = cfg

    def train_step(self, model, criterion, batch):
        loss = criterion(model, batch)
        return loss
    
    @staticmethod
    def features2probs(model, reference_features, normal_image_features, rgb_image_features):   
        # geo_logit = torch.diag(torch.einsum('bd,cd->bc', reference_features, normal_image_features))
        # geo_score = model.geo_logit_proj(geo_logit.unsqueeze(1))
        # import ipdb; ipdb.set_trace()
        geo_score = model.geo_logit_proj(torch.cat([reference_features, normal_image_features], dim=-1))
        geo_detail_score = model.geo_detail_logit_proj(normal_image_features)
        texture_score = model.texture_logit_proj(rgb_image_features)
        geo_texture_score = model.geo_texture_logit_proj(torch.cat([normal_image_features, rgb_image_features], dim=-1))
        align_score = model.alignment_logit_proj(torch.cat([reference_features, rgb_image_features], dim=-1))
        # geo_texture_logit = torch.diag(torch.einsum('bd,cd->bc', normal_image_features, rgb_image_features))
        # geo_texture_score = model.geo_texture_logit_proj(geo_texture_logit.unsqueeze(1))
        # align_logit = torch.diag(torch.einsum('bd,cd->bc', reference_features, rgb_image_features))
        # align_score = model.alignment_logit_proj(align_logit.unsqueeze(1))

        return torch.cat([geo_score, geo_detail_score, texture_score, geo_texture_score, align_score], dim=1)
    
    @torch.no_grad()
    def valid_step(self, model, criterion, batch):
        reference_features = criterion.get_inference_features(
            model,
            batch[self.cfg.reference_type_column_name],
            batch[self.cfg.reference_input_column_name]
        )
        normal_image_features, rgb_image_features = criterion.get_image_features(
            model,
            batch[self.cfg.normal_pixels_column_name],
            batch[self.cfg.rgb_pixels_column_name]
        )

        probs = self.features2probs(model, reference_features, normal_image_features, rgb_image_features)
        return probs

    @staticmethod
    def pixel_values_to_pil_images(pixel_values):
        images = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = numpy_to_pil(images)
        return images

    def run_inference(self, model, criterion, dataloader):
        eval_dict = collections.defaultdict(list)
        logger.info("Running clip score...")
        for batch in dataloader:
            preds = self.valid_step(model, criterion, batch)
            labels = batch[self.cfg.label_column_name]
            loss = torch.nn.functional.mse_loss(preds, labels)

            eval_dict["reference_type_column_name"] += batch[self.cfg.reference_type_column_name]
            eval_dict["reference_idx_column_name"] += batch[self.cfg.reference_idx_column_name].tolist()
            eval_dict["normal_image"] += self.pixel_values_to_pil_images(batch[self.cfg.normal_pixels_column_name])
            eval_dict["rgb_image"] += self.pixel_values_to_pil_images(batch[self.cfg.rgb_pixels_column_name])

            eval_dict["preds"] += preds.tolist()
            eval_dict["labels"] += labels.tolist()
            eval_dict["loss"] += [loss]
            eval_dict["is_correct"] += (preds == labels).tolist()

        return eval_dict

    @torch.no_grad()
    def evaluate(self, model, criterion, dataloader):
        eval_dict = self.run_inference(model, criterion, dataloader)
        eval_dict = self.gather_dict(eval_dict)
        metrics = {
            "loss": (sum(eval_dict["loss"])/len(eval_dict["loss"])).tolist(),
            "accuracy": (np.sum(eval_dict["is_correct"], 0)/len(eval_dict["is_correct"])).tolist(),
            "num_samples": len(eval_dict["is_correct"])
        }
        if LoggerType.WANDB == self.accelerator.cfg.log_with:
            self.log_to_wandb(eval_dict)
        return metrics
