import os
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoProcessor
from trainer.configs.configs import instantiate_with_cfg

# load model
device = "cuda"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
resume_dir = "checkpoint/3dgen-score-mvclip-v1"
EVAL_DIMS = ["geometry", "geo_detail", "texture", "geo_texture", "alignment"]

processor = AutoProcessor.from_pretrained(processor_name_or_path)

def calc_probs(model, 
            reference, reference_type, 
            normal_image_0, normal_image_1, rgb_image_0, rgb_image_1):
    torch.no_grad()
    # preprocess
    assert reference_type in ["text", "image"], "Invalid reference type"
    if reference_type == "text":
        reference_inputs = processor(
                                text=reference,
                                padding=True,
                                truncation=True,
                                max_length=77,
                                return_tensors="pt",
                            ).to(device)
        reference_features, = model.get_text_features(**reference_inputs)
    else:
        reference_inputs = processor(
                                images=reference,
                                padding=True,
                                truncation=True,
                                max_length=77,
                                return_tensors="pt",
                            ).to(device)
        reference_features, = model.get_image_features(**reference_inputs)
    
    all_normal_image_inputs = processor(
                                images=[normal_image_0, normal_image_1],
                                padding=True,
                                truncation=True,
                                max_length=77,
                                return_tensors="pt",
                            ).to(device)
    all_rgb_image_inputs = processor(
                            images=[rgb_image_0, rgb_image_1],
                            padding=True,
                            truncation=True,
                            max_length=77,
                            return_tensors="pt",
                        ).to(device)

    all_normal_image_features, = model.get_multi_view_normal_features(**all_normal_image_inputs)
    all_rgb_image_features, = model.get_multi_view_rgb_features(**all_rgb_image_inputs)
    normal_image_0_features, normal_image_1_features = all_normal_image_features.chunk(2, dim=0)
    rgb_image_0_features, rgb_image_1_features = all_rgb_image_features.chunk(2, dim=0)
    
    # scores
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
    
    image_0_scores = torch.cat([geo_image_0_scores, geo_detail_image_0_scores, texture_image_0_scores, geo_texture_image_0_scores, align_image_0_scores], dim=0)
    image_1_scores = torch.cat([geo_image_1_scores, geo_detail_image_1_scores, texture_image_1_scores, geo_texture_image_1_scores, align_image_1_scores], dim=0)
    scores = torch.stack([image_0_scores, image_1_scores], dim=-1)
    probs = torch.softmax(scores, dim=-1)
    # image_0_probs, image_1_probs = probs[:, 0], probs[:, 1]
    return probs.cpu().tolist()

print(f"Loading model...")
cfg_path = os.path.join(resume_dir, "config.yaml")
ckpt_path = os.path.join(resume_dir, "model.pkl")

cfg = OmegaConf.load(cfg_path)
model = instantiate_with_cfg(cfg.model)

model.load_state_dict(torch.load(ckpt_path), strict=False)
model.eval()
model.to(device)
print(f"Loaded checkpoint from {ckpt_path}")

## text-to-3D
print("=====text-to-3D=====")
normal_images_0 = "src/example/198_mv_normal.png"
rgb_images_0 = "src/example/198_mv_rgb.png"
normal_images_1 = "src/example/198_ln_normal.png"
rgb_images_1 = "src/example/198_ln_rgb.png"
prompt = "A pair of tortoiseshell eyeglasses"

probs = calc_probs(model, prompt, "text", 
                    Image.open(normal_images_0), Image.open(normal_images_1),
                    Image.open(rgb_images_0), Image.open(rgb_images_1))
for i, prob in enumerate(probs):
    print(EVAL_DIMS[i], ": ", prob)

## image-to-3D
print("=====image-to-3D=====")
normal_images_0 = "src/example/319_lgm_normal.png"
rgb_images_0 = "src/example/319_lgm_rgb.png"
normal_images_1 = "src/example/319_sz123_normal.png"
rgb_images_1 = "src/example/319_sz123_rgb.png"
prompt = "src/example/319.png"

probs = calc_probs(model, Image.open(prompt), "image", 
                    Image.open(normal_images_0), Image.open(normal_images_1),
                    Image.open(rgb_images_0), Image.open(rgb_images_1))
for i, prob in enumerate(probs):
    print(EVAL_DIMS[i], ": ", prob)