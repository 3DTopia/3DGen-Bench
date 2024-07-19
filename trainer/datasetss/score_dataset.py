import os
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, List

import random
import torch
from PIL import Image
from accelerate.logging import get_logger
# from datasets import load_from_disk, load_dataset, Dataset
from hydra.utils import instantiate
from omegaconf import II

from trainer.datasetss.base_dataset import BaseDataset, BaseDatasetConfig

logger = get_logger(__name__)
EVAL_DIMS = ["geometry", "geo_detail", "texture", "geo_texture", "alignment"]

def simple_collate(batch, column_name):
    return torch.cat([item[column_name] for item in batch], dim=0)

@dataclass
class ProcessorConfig:
    _target_: str = "transformers.AutoProcessor.from_pretrained"
    pretrained_model_name_or_path: str = II("model.mv_clip.pretrained_clip_model_name_or_path")


@dataclass
class ScoreDatasetConfig(BaseDatasetConfig):
    _target_: str = "trainer.datasetss.score_dataset.ScoreDataset"
    records_dir: str = "data/score_annotations"
    gallery_dir: str = "data/gallery"
    text_gallery_path: str = os.path.join(gallery_dir, "prompts_510.json")
    image_gallery_dir: str = os.path.join(gallery_dir, "rgba")
    con_images_dir: str = "data/surrounding_views"
    split_con_image: bool = False

    train_split_name: str = "train"
    valid_split_name: str = "valid"
    test_split_name: str = "test"
    cache_dir: Optional[str] = None

    reference_type_column_name: str = "reference_type"
    reference_idx_column_name: str = "reference_idx"
    reference_column_name: str = "reference"
    normal_image_column_name: str = "normal_image"
    rgb_image_column_name: str = "rgb_image"
    eval_dims_column_name: str = "eval_dims" 
    label_column_name: str = "label"
    are_different_column_name: str = "are_different"
    has_label_column_name: str = "has_label"
    
    reference_input_column_name: str = "reference_input"
    normal_pixels_column_name: str = "normal_pixel_values"
    rgb_pixels_column_name: str = "rgb_pixel_values"

    num_examples_per_prompt_column_name: str = "num_example_per_prompt"

    shuffle: bool = True
    keep_only_different: bool = False
    keep_only_with_label: bool = False
    keep_only_with_label_in_non_train: bool = True

    processor: ProcessorConfig = ProcessorConfig()

    limit_examples_per_prompt: int = -1

    only_on_best: bool = False


class ScoreDataset(BaseDataset):
    def __init__(self, cfg: ScoreDatasetConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.eval_dims = [f"{dim}_score" for dim in EVAL_DIMS]
        
        assert os.path.isfile(cfg.text_gallery_path), "Invalid text gallery path"
        with open(cfg.text_gallery_path, encoding="utf-8") as f:
            self.text_gallery_list = json.load(f)

        assert os.path.isdir(cfg.image_gallery_dir), f"Invalid image gallery dir"
        self.image_gallery_dir = cfg.image_gallery_dir

        assert os.path.isdir(cfg.con_images_dir), f"Invalid multi-view image dir"
        self.con_images_dir = cfg.con_images_dir

        logger.info(f"Loading {self.split} dataset")
        record_path = os.path.join(cfg.records_dir, f"{split}.json")
        with open(record_path, encoding='utf-8') as f:
            records = json.load(f)
        
        self.dataset = []
        for name, scores in records.items():
            task, model, offline_idx = name.split('_')
            offline_idx = int(offline_idx)
            if task=="text2shape":
                reference_type = "text"
                reference = self.text_gallery_list[offline_idx]
            elif task=="image2shape":
                reference_type = "image"
                reference = os.path.join(self.image_gallery_dir, f"{offline_idx}.png")
            else:
                continue
            
            data = {
                cfg.reference_type_column_name: reference_type,
                cfg.reference_idx_column_name: offline_idx,
                cfg.reference_column_name: reference,
                cfg.normal_image_column_name: os.path.join(self.con_images_dir, task, model, f"{offline_idx}_normal.png"),
                cfg.rgb_image_column_name: os.path.join(self.con_images_dir, task, model, f"{offline_idx}_rgb.png"),
                cfg.eval_dims_column_name: self.eval_dims,
                cfg.label_column_name: [scores[dim] for dim in self.eval_dims],
            }
            self.dataset.append(data)
        
        if self.cfg.shuffle:
            random.shuffle(self.dataset)
        logger.info(f"Loaded {len(self.dataset)} examples from {self.split} dataset")

        processor = instantiate(cfg.processor)
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor


    def process_text(self, text):
        input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids

    def process_image(self, image):
        if isinstance(image, dict):
            image = image["bytes"]
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image))
        else:
            image = Image.open(image)
        image = image.convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values
    
    def process_reference(self, example):
        if example[self.cfg.reference_type_column_name] == "text":
            return self.process_text(example[self.cfg.reference_column_name])
        elif example[self.cfg.reference_type_column_name] == "image":
            return self.process_image(example[self.cfg.reference_column_name])

    def __getitem__(self, idx):
        example = self.dataset[idx]

        reference_input = self.process_reference(example)

        normal_pixel_values = self.process_image(example[self.cfg.normal_image_column_name])
        rgb_pixel_values = self.process_image(example[self.cfg.rgb_image_column_name])

        item = {
            self.cfg.reference_type_column_name: example[self.cfg.reference_type_column_name],
            self.cfg.reference_idx_column_name: example[self.cfg.reference_idx_column_name],
            self.cfg.reference_input_column_name: reference_input,
            self.cfg.normal_pixels_column_name: normal_pixel_values,
            self.cfg.rgb_pixels_column_name: rgb_pixel_values,
            self.cfg.label_column_name: torch.tensor(example[self.cfg.label_column_name])[None],
            # self.cfg.num_examples_per_prompt_column_name: torch.tensor(example[self.cfg.num_examples_per_prompt_column_name])[None],
        }
        return item

    def collate_fn(self, batch):
        reference_types = [example[self.cfg.reference_type_column_name] for example in batch]
        reference_inputs = [example[self.cfg.reference_input_column_name] for example in batch]
        reference_indices = torch.tensor([example[self.cfg.reference_idx_column_name] for example in batch])
        normal_pixel_values = simple_collate(batch, self.cfg.normal_pixels_column_name)
        rgb_pixel_values = simple_collate(batch, self.cfg.rgb_pixels_column_name)
        labels = simple_collate(batch, self.cfg.label_column_name)
        # num_examples_per_prompt = simple_collate(batch, self.cfg.num_examples_per_prompt_column_name)

        normal_pixel_values = normal_pixel_values.to(memory_format=torch.contiguous_format).float()
        rgb_pixel_values = rgb_pixel_values.to(memory_format=torch.contiguous_format).float()

        collated = {
            self.cfg.reference_type_column_name: reference_types,
            self.cfg.reference_idx_column_name: reference_indices,
            self.cfg.reference_input_column_name: reference_inputs,
            self.cfg.normal_pixels_column_name: normal_pixel_values,
            self.cfg.rgb_pixels_column_name: rgb_pixel_values,
            self.cfg.label_column_name: labels,
            # self.cfg.num_examples_per_prompt_column_name: num_examples_per_prompt,
        }
        return collated

    def __len__(self):
        return len(self.dataset)
