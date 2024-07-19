import os
import copy
import torch
import torch.nn as nn
from collections import OrderedDict

from typing import List, Union
from omegaconf import II
from dataclasses import dataclass, field
from trainer.models.base_model import BaseModelConfig
from trainer.models.mvclip_model import MVCLIPModelConfig, MVCLIPModel


@dataclass
class ScoreModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.score_model.ScoreModel"
    pretrained_mvclip_model_name_or_path: str = "checkpoint-final"
    mv_clip: dict = field(default_factory=lambda: {
        "_target_": "trainer.models.mvclip_model.MVCLIPModel",
        "pretrained_clip_model_name_or_path": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "freeze_clip": True,
        "freeze_vision": True,
        "freeze_logit_scale": True,
        "freeze_logit_proj": True
    })
    freeze_mvclip: bool = True
    resume_mvclip_proj: bool = True


class ScoreModel(nn.Module):
    def __init__(self, cfg: ScoreModelConfig):
        super().__init__()
        self.cfg = cfg
        # prompt model
        self.mvclip = MVCLIPModel(MVCLIPModelConfig(**self.cfg.mv_clip))
        self.load_mvclip_from_ckpt()

        # project model
        self.h_dim = int(self.mvclip.rgb_visual_projection.out_features)
        self.geo_logit_proj = nn.Linear(self.h_dim*2, 1)
        self.geo_detail_logit_proj = nn.Linear(self.h_dim, 1)
        self.texture_logit_proj = nn.Linear(self.h_dim, 1)
        self.geo_texture_logit_proj = nn.Linear(self.h_dim*2, 1)
        self.alignment_logit_proj = nn.Linear(self.h_dim*2, 1)
        # initialize
        if self.cfg.resume_mvclip_proj:
            print(f"Initialze from mvclip")
            self.geo_logit_proj.weight.data.fill_(self.mvclip.geo_logit_scale)
            self.geo_logit_proj.bias.data.fill_(0)
            self.geo_detail_logit_proj = copy.deepcopy(self.mvclip.geo_detail_logit_proj)
            self.texture_logit_proj = copy.deepcopy(self.mvclip.texture_logit_proj)
            self.geo_texture_logit_proj.weight.data.fill_(self.mvclip.geo_texture_logit_scale)
            self.geo_texture_logit_proj.bias.data.fill_(0)
            self.alignment_logit_proj.weight.data.fill_(self.mvclip.align_logit_scale)
            self.alignment_logit_proj.bias.data.fill_(0)
        else:
            print(f"Initialize from scratch")
            projs = [self.geo_logit_proj, self.geo_detail_logit_proj, self.texture_logit_proj, self.geo_texture_logit_proj, self.alignment_logit_proj]
            for proj_func in projs:
                nn.init.xavier_uniform(proj_func.weight)
                proj_func.bias.data.fill_(0)
        
        ## freeze params
        if self.cfg.freeze_mvclip:
            for param in self.mvclip.parameters():
                param.requires_grad = False
        
    def load_mvclip_from_ckpt(self, ckpt_path_or_name=None, strict=False):
        ckpt_path_or_name = ckpt_path_or_name if ckpt_path_or_name else self.cfg.pretrained_mvclip_model_name_or_path
        if os.path.isdir(ckpt_path_or_name):
            ckpt_path = os.path.join(ckpt_path_or_name, "model.pkl")
        else:
            ckpt_path = ckpt_path_or_name

        print(f"Loading MVCLIPModel from {ckpt_path}")
        self.mvclip.load_state_dict(torch.load(ckpt_path), strict=strict)

    def get_text_features(self, *args, **kwargs):
        return self.mvclip.get_text_features(*args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        return self.mvclip.get_image_features(*args, **kwargs)

    def get_multi_view_normal_features(self, *args, **kwargs):
        return self.mvclip.get_multi_view_normal_features( *args, **kwargs)
    
    def get_multi_view_rgb_features(self, *args, **kwargs):
        return self.mvclip.get_multi_view_rgb_features(*args, **kwargs)
    

    def forward(self, inference_intputs=None, inference_type=None, normal_images=None, rgb_images=None, return_dict=False):
        inference_features, normal_features, rgb_features = self.mvclip(inference_intputs=inference_intputs, inference_type=inference_type, 
                                                                        normal_images=normal_images, rgb_images=rgb_images)

        geo_score = self.geo_logit_proj(torch.cat([inference_features, normal_features], dim=-1)).squeeze(1)
        geo_detail_score = self.geo_detail_logit_proj(normal_features).squeeze(1)
        texture_score = self.texture_logit_proj(rgb_features).squeeze(1)
        geo_texture_score = self.geo_texture_logit_proj(torch.cat([normal_features, rgb_features], dim=-1)).squeeze(1)
        align_score = self.alignment_logit_proj(torch.cat([inference_features, rgb_features], dim=-1)).squeeze(1)
        
        if return_dict:
            return {
                "geo": geo_score,
                "geo_detail": geo_detail_score,
                "texture_score": texture_score,
                "geo_texture_score": geo_texture_score,
                "align_score": align_score
                }
        else:
            return geo_score, geo_detail_score, texture_score, geo_texture_score, align_score
    

    # def forward(self, text_inputs=None, image_inputs=None):
    #     outputs = ()
    #     if text_inputs is not None:
    #         outputs += self.get_text_features(text_inputs),
    #     if image_inputs is not None:
    #         outputs += self.get_image_features(image_inputs),
    #     return outputs

    def save(self, path):
        torch.save(self.state_dict(), path)
    