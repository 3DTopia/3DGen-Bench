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
    pretrained_model_name_or_path: str = II("model.pretrained_clip_model_name_or_path")


@dataclass
class MVDatasetConfig(BaseDatasetConfig):
    _target_: str = "trainer.datasetss.dataset.MVDataset"
    records_dir: str = "data/preference_annotation"
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
    normal_image_0_column_name: str = "normal_image_0"
    normal_image_1_column_name: str = "normal_image_1"
    rgb_image_0_column_name: str = "rgb_image_0"
    rgb_image_1_column_name: str = "rgb_image_1"
    eval_dims_column_name: str = "eval_dims" 
    label_0_column_name: str = "label_0"
    label_1_column_name: str = "label_1"
    are_different_column_name: str = "are_different"
    has_label_column_name: str = "has_label"
    
    reference_input_column_name: str = "reference_input"
    normal_pixels_0_column_name: str = "normal_pixel_values_0"
    normal_pixels_1_column_name: str = "normal_pixel_values_1"
    rgb_pixels_0_column_name: str = "rgb_pixel_values_0"
    rgb_pixels_1_column_name: str = "rgb_pixel_values_1"

    num_examples_per_prompt_column_name: str = "num_example_per_prompt"

    shuffle: bool = True
    keep_only_different: bool = False
    keep_only_with_label: bool = False
    keep_only_with_label_in_non_train: bool = True

    processor: ProcessorConfig = ProcessorConfig()

    limit_examples_per_prompt: int = -1

    only_on_best: bool = False


class MVDataset(BaseDataset):
    def __init__(self, cfg: MVDatasetConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.eval_dims = EVAL_DIMS
        
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
        for name, record in records.items():
            if record["task"]=="text2shape":
                reference_type = "text"
                offline_idx = int(record["offline_idx"])
                reference = self.text_gallery_list[offline_idx]
            elif record["task"]=="image2shape":
                reference_type = "image"
                offline_idx = int(record["offline_idx"])
                reference = os.path.join(self.image_gallery_dir, f"{offline_idx}.png")
            else:
                continue
            
            data = {
                cfg.reference_type_column_name: reference_type,
                cfg.reference_idx_column_name: offline_idx,
                cfg.reference_column_name: reference,
                cfg.normal_image_0_column_name: os.path.join(self.con_images_dir, record["task"], record["models"][0], f"{offline_idx}_normal.png"),
                cfg.normal_image_1_column_name: os.path.join(self.con_images_dir, record["task"], record["models"][1], f"{offline_idx}_normal.png"),
                cfg.rgb_image_0_column_name: os.path.join(self.con_images_dir, record["task"], record["models"][0], f"{offline_idx}_rgb.png"),
                cfg.rgb_image_1_column_name: os.path.join(self.con_images_dir, record["task"], record["models"][1], f"{offline_idx}_rgb.png"),
                cfg.eval_dims_column_name: self.eval_dims,
                cfg.label_0_column_name: [record[dim]["label_0"] for dim in self.eval_dims],
                cfg.label_1_column_name: [record[dim]["label_1"] for dim in self.eval_dims],
                # cfg.num_examples_per_prompt_column_name: 
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

        normal_pixel_0_values = self.process_image(example[self.cfg.normal_image_0_column_name])
        normal_pixel_1_values = self.process_image(example[self.cfg.normal_image_1_column_name])
        rgb_pixel_0_values = self.process_image(example[self.cfg.rgb_image_0_column_name])
        rgb_pixel_1_values = self.process_image(example[self.cfg.rgb_image_1_column_name])

        item = {
            self.cfg.reference_type_column_name: example[self.cfg.reference_type_column_name],
            self.cfg.reference_idx_column_name: example[self.cfg.reference_idx_column_name],
            self.cfg.reference_input_column_name: reference_input,
            self.cfg.normal_pixels_0_column_name: normal_pixel_0_values,
            self.cfg.normal_pixels_1_column_name: normal_pixel_1_values,
            self.cfg.rgb_pixels_0_column_name: rgb_pixel_0_values,
            self.cfg.rgb_pixels_1_column_name: rgb_pixel_1_values,
            self.cfg.label_0_column_name: torch.tensor(example[self.cfg.label_0_column_name])[None],
            self.cfg.label_1_column_name: torch.tensor(example[self.cfg.label_1_column_name])[None],
            # self.cfg.num_examples_per_prompt_column_name: torch.tensor(example[self.cfg.num_examples_per_prompt_column_name])[None],
        }
        return item

    def collate_fn(self, batch):
        reference_types = [example[self.cfg.reference_type_column_name] for example in batch]
        reference_inputs = [example[self.cfg.reference_input_column_name] for example in batch]  # simple_collate(batch, self.cfg.reference_input_column_name)
        reference_indices = torch.tensor([example[self.cfg.reference_idx_column_name] for example in batch])
        normal_pixel_0_values = simple_collate(batch, self.cfg.normal_pixels_0_column_name)
        normal_pixel_1_values = simple_collate(batch, self.cfg.normal_pixels_1_column_name)
        rgb_pixel_0_values = simple_collate(batch, self.cfg.rgb_pixels_0_column_name)
        rgb_pixel_1_values = simple_collate(batch, self.cfg.rgb_pixels_1_column_name)
        label_0 = simple_collate(batch, self.cfg.label_0_column_name)
        label_1 = simple_collate(batch, self.cfg.label_1_column_name)
        # num_examples_per_prompt = simple_collate(batch, self.cfg.num_examples_per_prompt_column_name)

        normal_pixel_0_values = normal_pixel_0_values.to(memory_format=torch.contiguous_format).float()
        normal_pixel_1_values = normal_pixel_1_values.to(memory_format=torch.contiguous_format).float()
        rgb_pixel_0_values = rgb_pixel_0_values.to(memory_format=torch.contiguous_format).float()
        rgb_pixel_1_values = rgb_pixel_1_values.to(memory_format=torch.contiguous_format).float()

        collated = {
            self.cfg.reference_type_column_name: reference_types,
            self.cfg.reference_idx_column_name: reference_indices,
            self.cfg.reference_input_column_name: reference_inputs,
            self.cfg.normal_pixels_0_column_name: normal_pixel_0_values,
            self.cfg.normal_pixels_1_column_name: normal_pixel_1_values,
            self.cfg.rgb_pixels_0_column_name: rgb_pixel_0_values,
            self.cfg.rgb_pixels_1_column_name: rgb_pixel_1_values,
            self.cfg.label_0_column_name: label_0,
            self.cfg.label_1_column_name: label_1,
            # self.cfg.num_examples_per_prompt_column_name: num_examples_per_prompt,
        }
        return collated

    def __len__(self):
        return len(self.dataset)
