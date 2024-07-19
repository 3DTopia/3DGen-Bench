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
class MVCLIPTaskConfig(BaseTaskConfig):
    _target_: str = "trainer.tasks.mvclip_task.MVCLIPTask"
    pretrained_clip_model_name_or_path: str = II("model.pretrained_clip_model_name_or_path")
    label_0_column_name: str = II("dataset.label_0_column_name")
    label_1_column_name: str = II("dataset.label_1_column_name")

    reference_type_column_name: str = II("dataset.reference_type_column_name")
    reference_input_column_name: str = II("dataset.reference_input_column_name")
    reference_idx_column_name: str = II("dataset.reference_idx_column_name")
    normal_pixels_0_column_name: str = II("dataset.normal_pixels_0_column_name")
    normal_pixels_1_column_name: str = II("dataset.normal_pixels_1_column_name")
    rgb_pixels_0_column_name: str = II("dataset.rgb_pixels_0_column_name")
    rgb_pixels_1_column_name: str = II("dataset.rgb_pixels_1_column_name")

def numpy_to_pil(images):
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


class MVCLIPTask(BaseTask):
    def __init__(self, cfg: MVCLIPTaskConfig, accelerator: BaseAccelerator):
        super().__init__(cfg, accelerator)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_clip_model_name_or_path)
        self.cfg = cfg

    def train_step(self, model, criterion, batch):
        loss = criterion(model, batch)
        return loss

    @staticmethod
    def features2probs(model, 
                       reference_features, 
                       normal_image_0_features, normal_image_1_features, 
                       rgb_image_0_features, rgb_image_1_features):
        geo_image_0_scores = model.geo_logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', reference_features, normal_image_0_features))
        geo_image_1_scores = model.geo_logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', reference_features, normal_image_1_features))
        
        geo_detail_image_0_scores = model.geo_detail_logit_proj(normal_image_0_features).squeeze(1)
        geo_detail_image_1_scores = model.geo_detail_logit_proj(normal_image_1_features).squeeze(1)
        
        texture_image_0_scores = model.texture_logit_proj(rgb_image_0_features).squeeze(1)
        texture_image_1_scores = model.texture_logit_proj(rgb_image_1_features).squeeze(1)
        
        geo_texture_image_0_scores = model.geo_texture_logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', normal_image_0_features, rgb_image_0_features))
        geo_texture_image_1_scores = model.geo_texture_logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', normal_image_1_features, rgb_image_1_features))
        
        align_image_0_scores = model.align_logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', reference_features, rgb_image_0_features))
        align_image_1_scores = model.align_logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', reference_features, rgb_image_1_features))
        
        # import ipdb; ipdb.set_trace()
        image_0_scores = torch.cat([geo_image_0_scores, geo_detail_image_0_scores, texture_image_0_scores, geo_texture_image_0_scores, align_image_0_scores], dim=0)
        image_1_scores = torch.cat([geo_image_1_scores, geo_detail_image_1_scores, texture_image_1_scores, geo_texture_image_1_scores, align_image_1_scores], dim=0)
        scores = torch.stack([image_0_scores, image_1_scores], dim=-1)
        probs = torch.softmax(scores, dim=-1)
        image_0_probs, image_1_probs = probs[:, 0], probs[:, 1]
        return image_0_probs, image_1_probs

    @torch.no_grad()
    def valid_step(self, model, criterion, batch):
        reference_features = criterion.get_inference_features(
            model,
            batch[self.cfg.reference_type_column_name],
            batch[self.cfg.reference_input_column_name]
        )
        normal_image_0_features, normal_image_1_features, rgb_image_0_features, rgb_image_1_features = criterion.get_image_features(
            model,
            batch[self.cfg.normal_pixels_0_column_name],
            batch[self.cfg.normal_pixels_1_column_name],
            batch[self.cfg.rgb_pixels_0_column_name],
            batch[self.cfg.rgb_pixels_1_column_name]
        )

        loss = criterion.calc_loss(
            model,
            reference_features,
            normal_image_0_features,
            normal_image_1_features,
            rgb_image_0_features,
            rgb_image_1_features,
            batch[self.cfg.label_0_column_name],
            batch[self.cfg.label_1_column_name],
            # batch[self.cfg.num_examples_per_prompt_column_name],
        )

        image_0_probs, image_1_probs = self.features2probs(model, reference_features, 
                                                           normal_image_0_features, normal_image_1_features, 
                                                           rgb_image_0_features, rgb_image_1_features)
        return loss, image_0_probs, image_1_probs

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
            loss, image_0_probs, image_1_probs = self.valid_step(model, criterion, batch)
            image_0_dim_probs = image_0_probs.chunk(len(EVAL_DIMS))
            image_1_dim_probs = image_1_probs.chunk(len(EVAL_DIMS))
            image_0_dim_probs = torch.cat([dim_probs.unsqueeze(1) for dim_probs in image_0_dim_probs], dim=1)
            image_1_dim_probs = torch.cat([dim_probs.unsqueeze(1) for dim_probs in image_1_dim_probs], dim=1)

            agree_on_0 = (image_0_dim_probs > image_1_dim_probs) * batch[self.cfg.label_0_column_name]
            agree_on_1 = (image_0_dim_probs < image_1_dim_probs) * batch[self.cfg.label_1_column_name]
            is_correct = agree_on_0 + agree_on_1
            eval_dict["is_correct"] += is_correct.tolist()
            eval_dict["reference_type_column_name"] += batch[self.cfg.reference_type_column_name]
            eval_dict["reference_idx_column_name"] += batch[self.cfg.reference_idx_column_name].tolist()
            # eval_dict["captions"] += self.tokenizer.batch_decode(
            #     batch[self.cfg.input_ids_column_name],
            #     skip_special_tokens=True
            # )
            eval_dict["normal_image_0"] += self.pixel_values_to_pil_images(batch[self.cfg.normal_pixels_0_column_name])
            eval_dict["normal_image_1"] += self.pixel_values_to_pil_images(batch[self.cfg.normal_pixels_1_column_name])
            eval_dict["rgb_image_0"] += self.pixel_values_to_pil_images(batch[self.cfg.rgb_pixels_0_column_name])
            eval_dict["rgb_image_1"] += self.pixel_values_to_pil_images(batch[self.cfg.rgb_pixels_1_column_name])
            eval_dict["prob_0"] += image_0_dim_probs.tolist()
            eval_dict["prob_1"] += image_1_dim_probs.tolist()

            eval_dict["label_0"] += batch[self.cfg.label_0_column_name].tolist()
            eval_dict["label_1"] += batch[self.cfg.label_1_column_name].tolist()
            eval_dict["loss"] += [loss]

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
