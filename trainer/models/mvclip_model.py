import copy
import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import CLIPModel as HFCLIPModel

from typing import List, Union
from omegaconf import II
from dataclasses import dataclass
from trainer.models.base_model import BaseModelConfig

@dataclass
class MVCLIPModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.mvclip_model.MVCLIPModel"
    pretrained_clip_model_name_or_path: str = "openai/clip-vit-base-patch32"
    # pretrained_proj_model_name_or_path: Union[str, None] = None
    freeze_clip: bool = True
    freeze_vision: bool = False
    freeze_logit_scale: bool = False
    freeze_logit_proj: bool = False


class MVCLIPModel(nn.Module):
    def __init__(self, cfg: MVCLIPModelConfig):
        super().__init__()
        self.cfg = cfg
        # prompt model
        self.clip = HFCLIPModel.from_pretrained(cfg.pretrained_clip_model_name_or_path)
        # vision model
        self.normal_vision_model = copy.deepcopy(self.clip.vision_model)
        self.normal_visual_projection = copy.deepcopy(self.clip.visual_projection)
        self.rgb_vision_model = copy.deepcopy(self.clip.vision_model)
        self.rgb_visual_projection = copy.deepcopy(self.clip.visual_projection)

        self.geo_logit_scale = copy.deepcopy(self.clip.logit_scale)
        self.geo_texture_logit_scale = copy.deepcopy(self.clip.logit_scale)
        self.align_logit_scale = copy.deepcopy(self.clip.logit_scale)
        # project model
        self.h_dim = self.rgb_visual_projection.out_features
        self.geo_detail_logit_proj = nn.Linear(self.h_dim, 1)
        self.texture_logit_proj = nn.Linear(self.h_dim, 1)
        # initialize
        # if self.cfg.pretrained_proj_model_name_or_path:
        nn.init.xavier_uniform(self.geo_detail_logit_proj.weight)
        self.geo_detail_logit_proj.bias.data.fill_(0)
        nn.init.xavier_uniform(self.texture_logit_proj.weight)
        self.texture_logit_proj.bias.data.fill_(0)
        
        ## freeze params
        if self.cfg.freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
        if self.cfg.freeze_vision:
            for module in [self.normal_vision_model, self.rgb_vision_model]:
                for param in module.parameters():
                    param.requires_grad = False
        if self.cfg.freeze_logit_scale:
            for scale in [self.geo_logit_scale, self.geo_texture_logit_scale, self.align_logit_scale]:
                scale.requires_grad = False
        if self.cfg.freeze_logit_proj:
            for proj in [self.geo_detail_logit_proj, self.texture_logit_proj]:
                for param in proj.parameters():
                    param.requires_grad = False
        

    def get_text_features(self, *args, **kwargs):
        return self.clip.get_text_features(*args, **kwargs),

    def get_image_features(self, *args, **kwargs):
        return self.clip.get_image_features(*args, **kwargs),

    def get_multi_view_normal_features(self, *args, **kwargs):
        vision_outputs = self.normal_vision_model(*args, **kwargs)
        pooled_output = vision_outputs[1]
        return self.normal_visual_projection(pooled_output),
    
    def get_multi_view_rgb_features(self, *args, **kwargs):
        vision_outputs = self.rgb_vision_model(*args, **kwargs)
        pooled_output = vision_outputs[1]
        return self.rgb_visual_projection(pooled_output),
    
    def forward(self, inference_intputs=None, inference_type=None, normal_images=None, rgb_images=None):
        outputs = ()
        if inference_intputs is not None:
            assert inference_type, "Empty inference type"
            assert inference_type in ["text", "image"], f"Invalid inference type {inference_type}"
            if inference_type == "text":
                outputs += self.get_text_features(inference_intputs)
            if inference_type == "image":
                outputs += self.get_image_features(inference_intputs)

        if normal_images is not None:
            outputs += self.get_multi_view_normal_features(normal_images)
        if rgb_images is not None:
            outputs += self.get_multi_view_rgb_features(rgb_images)
        
        return outputs
    
    def cal_logits(self, inference_intputs=None, inference_type=None, normal_images=None, rgb_images=None, return_dict=False):
        inference_features, normal_features, rgb_features = self.forward(inference_intputs=inference_intputs, inference_type=inference_type, 
                                                                         normal_images=normal_images, rgb_images=rgb_images)
        geo_logit = self.geo_logit_scale * normal_features @ inference_features.T
        geo_detail_logit = self.geo_detail_logit_proj(normal_features)
        texture_logit = self.texture_logit_proj(rgb_features)
        geo_texture_logit = self.geo_texture_logit_scale * normal_features @ rgb_features.T
        align_logit = self.align_logit_scale * rgb_features @ inference_features.T

        if return_dict:
            return {
                "geometry": geo_logit,
                "geo_detail": geo_detail_logit,
                "texture": texture_logit,
                "geo_texture": geo_texture_logit,
                "alignmnet": align_logit,
            }
        else:
            return (geo_logit, geo_detail_logit, texture_logit, geo_texture_logit, align_logit)

    def save(self, path):
        torch.save(self.state_dict(), path)
    