from dataclasses import dataclass
import torch
from omegaconf import II
from torch.nn.modules.loss import _Loss


@dataclass
class ScoreCriterionConfig:
    _target_: str = "trainer.criterions.score_criterion.ScoreCriterion"
    is_distributed: bool = True
    reference_type_column_name: str = II("dataset.reference_type_column_name")
    reference_input_column_name: str = II("dataset.reference_input_column_name")

    label_column_name: str = II("dataset.label_column_name")
    normal_pixels_column_name: str = II("dataset.normal_pixels_column_name")
    rgb_pixels_column_name: str = II("dataset.rgb_pixels_column_name")
    num_examples_per_prompt_column_name: str = II("dataset.num_examples_per_prompt_column_name")
    pass


class ScoreCriterion(_Loss):
    def __init__(self, cfg: ScoreCriterionConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def get_inference_features(model, reference_types, reference_inputs):
        text_idxs = [idx for idx, type in enumerate(reference_types) if type=="text"]
        image_idxs = [idx for idx, type in enumerate(reference_types) if type=="image"]
        text_reference_inputs = torch.cat([reference_inputs[i] for i in text_idxs], dim=0) if len(text_idxs)>0 else None
        image_reference_inputs = torch.cat([reference_inputs[i] for i in image_idxs], dim=0) if len(image_idxs)>0 else None

        if text_reference_inputs == None:
            image_reference_features, = model.get_image_features(image_reference_inputs)
            image_reference_features = image_reference_features / image_reference_features.norm(dim=-1, keepdim=True)
            reference_features = image_reference_features
        elif image_reference_inputs == None:
            text_reference_features, = model.get_text_features(text_reference_inputs)
            text_reference_features = text_reference_features / text_reference_features.norm(dim=-1, keepdim=True)
            reference_features = text_reference_features
        else:
            text_reference_features, = model.get_text_features(text_reference_inputs)
            image_reference_features, =  model.get_image_features(image_reference_inputs)
            text_reference_features = text_reference_features / text_reference_features.norm(dim=-1, keepdim=True)
            image_reference_features = image_reference_features / image_reference_features.norm(dim=-1, keepdim=True)
            reference_features = torch.cat([text_reference_features, image_reference_features], dim=0)
            reference_features[text_idxs] = text_reference_features
            reference_features[image_idxs] = image_reference_features


        return reference_features


    @staticmethod
    def get_image_features(model, normal_pixels_values, rgb_pixels_values):
        normal_image_features, = model.get_multi_view_normal_features(normal_pixels_values)
        rgb_image_features, = model.get_multi_view_rgb_features(rgb_pixels_values)
        return normal_image_features, rgb_image_features

    @staticmethod
    def gather_features(features):
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        return all_features

    def calc_loss(
            self,
            model,
            reference_features, normal_image_features, rgb_image_features,
            labels,
            # num_examples_per_prompt,
            *args,
            **kwargs
    ):
        device = model.device

        # gather features
        if self.cfg.is_distributed:
            reference_features = self.gather_features(reference_features)
            normal_image_features = self.gather_features(normal_image_features)
            rgb_image_features = self.gather_features(rgb_image_features)
            labels = self.gather_features(labels)
            # num_examples_per_prompt = self.gather_features(num_examples_per_prompt)

        # calc logits
        geo_score = model.geo_logit_proj(torch.cat([reference_features, normal_image_features], dim=-1))
        geo_detail_score = model.geo_detail_logit_proj(normal_image_features)
        texture_score = model.texture_logit_proj(rgb_image_features)
        geo_texture_score = model.geo_texture_logit_proj(torch.cat([normal_image_features, rgb_image_features], dim=-1))
        align_score = model.alignment_logit_proj(torch.cat([reference_features, rgb_image_features], dim=-1))
        # concate
        scores = torch.stack([geo_score, geo_detail_score, texture_score, geo_texture_score, align_score], dim=1).squeeze(-1)

        eval_loss = torch.nn.functional.mse_loss(scores, labels)
        loss = eval_loss.sum()
        return loss

    def forward(self, model, batch):
        reference_features = self.get_inference_features(
            model,
            batch[self.cfg.reference_type_column_name],
            batch[self.cfg.reference_input_column_name]
        )
        normal_image_features, rgb_image_features = self.get_image_features(
            model,
            batch[self.cfg.normal_pixels_column_name],
            batch[self.cfg.rgb_pixels_column_name],
        )

        loss = self.calc_loss(
            model,
            reference_features,
            normal_image_features, rgb_image_features,
            batch[self.cfg.label_column_name],
            # batch[self.cfg.num_examples_per_prompt_column_name],
        )
        return loss
