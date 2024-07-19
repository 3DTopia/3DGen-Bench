from dataclasses import dataclass
import torch
from omegaconf import II
from torch.nn.modules.loss import _Loss


@dataclass
class MVCriterionConfig:
    _target_: str = "trainer.criterions.criterion.MVCriterion"
    is_distributed: bool = True
    reference_type_column_name: str = II("dataset.reference_type_column_name")
    reference_input_column_name: str = II("dataset.reference_input_column_name")

    label_0_column_name: str = II("dataset.label_0_column_name")
    label_1_column_name: str = II("dataset.label_1_column_name")
    normal_pixels_0_column_name: str = II("dataset.normal_pixels_0_column_name")
    normal_pixels_1_column_name: str = II("dataset.normal_pixels_1_column_name")
    rgb_pixels_0_column_name: str = II("dataset.rgb_pixels_0_column_name")
    rgb_pixels_1_column_name: str = II("dataset.rgb_pixels_1_column_name")
    num_examples_per_prompt_column_name: str = II("dataset.num_examples_per_prompt_column_name")
    in_batch_negatives: bool = False
    reward_modeling: bool = False
    pass


class MVCriterion(_Loss):
    def __init__(self, cfg: MVCriterionConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def get_inference_features(model, reference_types, reference_inputs):
        text_idxs = [idx for idx, type in enumerate(reference_types) if type=="text"]
        image_idxs = [idx for idx, type in enumerate(reference_types) if type=="image"]
        text_reference_inputs = torch.cat([reference_inputs[i] for i in text_idxs], dim=0) if len(text_idxs)>0 else None
        image_reference_inputs = torch.cat([reference_inputs[i] for i in image_idxs], dim=0) if len(image_idxs)>0 else None

        if text_reference_inputs == None:
            image_reference_features, = model(inference_intputs=image_reference_inputs, inference_type='image')
            image_reference_features = image_reference_features / image_reference_features.norm(dim=-1, keepdim=True)
            reference_features = image_reference_features
        elif image_reference_inputs == None:
            text_reference_features, = model(inference_intputs=text_reference_inputs, inference_type='text')
            text_reference_features = text_reference_features / text_reference_features.norm(dim=-1, keepdim=True)
            reference_features = text_reference_features
        else:
            text_reference_features, = model(inference_intputs=text_reference_inputs, inference_type='text')
            image_reference_features, = model(inference_intputs=image_reference_inputs, inference_type='image')
            text_reference_features = text_reference_features / text_reference_features.norm(dim=-1, keepdim=True)
            image_reference_features = image_reference_features / image_reference_features.norm(dim=-1, keepdim=True)
            reference_features = torch.cat([text_reference_features, image_reference_features], dim=0)
            reference_features[text_idxs] = text_reference_features
            reference_features[image_idxs] = image_reference_features

        return reference_features


    @staticmethod
    def get_image_features(model, 
                           normal_pixels_0_values, normal_pixels_1_values, 
                           rgb_pixels_0_values, rgb_pixels_1_values):
        all_normal_pixel_values = torch.cat([normal_pixels_0_values, normal_pixels_1_values], dim=0)
        all_rgb_pixel_values = torch.cat([rgb_pixels_0_values, rgb_pixels_1_values], dim=0)
        all_normal_image_features, all_rgb_image_features = model(normal_images=all_normal_pixel_values, rgb_images=all_rgb_pixel_values)
        all_normal_image_features = all_normal_image_features / all_normal_image_features.norm(dim=-1, keepdim=True)
        all_rgb_image_features = all_rgb_image_features / all_rgb_image_features.norm(dim=-1, keepdim=True)

        normal_image_0_features, normal_image_1_features = all_normal_image_features.chunk(2, dim=0)
        rgb_image_0_features, rgb_image_1_features = all_rgb_image_features.chunk(2, dim=0)
        return normal_image_0_features, normal_image_1_features, rgb_image_0_features, rgb_image_1_features

    @staticmethod
    def gather_features(features):
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        return all_features

    def calc_loss(
            self,
            model,
            reference_features,
            normal_image_0_features, normal_image_1_features,
            rgb_image_0_features, rgb_image_1_features,
            label_0,
            label_1,
            # num_examples_per_prompt,
            *args,
            **kwargs
    ):
        device = model.device

        # gather features
        if self.cfg.is_distributed:
            reference_features = self.gather_features(reference_features)
            normal_image_0_features = self.gather_features(normal_image_0_features)
            normal_image_1_features = self.gather_features(normal_image_1_features)
            rgb_image_0_features = self.gather_features(rgb_image_0_features)
            rgb_image_1_features = self.gather_features(rgb_image_1_features)
            label_0 = self.gather_features(label_0)
            label_1 = self.gather_features(label_1)
            # num_examples_per_prompt = self.gather_features(num_examples_per_prompt)

        # calc logits
        all_normal_image_features = torch.cat([normal_image_0_features, normal_image_1_features], dim=0)  # (2 * batch_size, dim)
        all_rgb_image_features = torch.cat([rgb_image_0_features, rgb_image_1_features], dim=0) 

        geo_logits_per_normal_image = model.geo_logit_scale * all_normal_image_features @ reference_features.T # (2*batch_size, batch_size)
        geo_logits_normal_image_0, geo_logits_normal_image_1 = geo_logits_per_normal_image.chunk(2, dim=0)  
        geo_logits_per_reference = model.geo_logit_scale * reference_features @ all_normal_image_features.T

        geo_detail_logits_per_normal_image = model.geo_detail_logit_proj(all_normal_image_features) # (2*batch_size, 1)
        geo_detail_logits_normal_image_0, geo_detail_logits_normal_image_1 = geo_detail_logits_per_normal_image.chunk(2, dim=0)
        
        texture_logits_per_rgb_image = model.texture_logit_proj(all_rgb_image_features) # (2*batch_size, 1)
        texture_logits_rgb_image_0, texture_logits_rgb_image_1 = texture_logits_per_rgb_image.chunk(2, dim=0)
        
        geo_texture_logits_0 = model.geo_texture_logit_scale * normal_image_0_features @ rgb_image_0_features.T # (batch_size, batch_size)
        geo_texture_logits_1 = model.geo_texture_logit_scale * normal_image_1_features @ rgb_image_1_features.T

        align_logits_per_rgb_image = model.align_logit_scale * all_rgb_image_features @ reference_features.T # (2*batch_size, batch_size)
        align_logits_rgb_image_0, align_logits_rgb_image_1 = align_logits_per_rgb_image.chunk(2, dim=0)
        align_logits_reference = model.align_logit_scale * reference_features @ all_rgb_image_features.T

        ## calc loss
        if self.cfg.in_batch_negatives:
            # get labels
            num_references = reference_features.shape[0]
            reference_labels = torch.arange(num_references, device=device, dtype=torch.long)
            num_normal_images, num_rgb_images = all_normal_image_features.shape[0], all_rgb_image_features.shape[0]
            normal_image_labels = torch.arange(num_normal_images, device=device, dtype=torch.long)
            rgb_image_labels = torch.arange(num_rgb_images, device=device, dtype=torch.long)
            normal_image_0_labels, normal_image_1_labels = normal_image_labels.chunk(2, dim=0)
            rgb_image_0_labels, rgb_image_1_labels = rgb_image_labels.chunk(2, dim=0)

            # we want to increase the logits of the preferred images to the reference
            ## normal
            normal_image_0_loss = torch.nn.functional.cross_entropy(geo_logits_normal_image_0, reference_labels, reduction="none")
            normal_image_1_loss = torch.nn.functional.cross_entropy(geo_logits_normal_image_1, reference_labels, reduction="none")
            normal_reference_0_loss = torch.nn.functional.cross_entropy(geo_logits_per_reference, normal_image_0_labels, reduction="none")
            normal_reference_1_loss = torch.nn.functional.cross_entropy(geo_logits_per_reference, normal_image_1_labels, reduction="none")
            
            ## rgb
            rgb_image_0_loss = torch.nn.functional.cross_entropy(align_logits_rgb_image_0, reference_labels, reduction="none")
            rgb_image_1_loss = torch.nn.functional.cross_entropy(align_logits_rgb_image_1, reference_labels, reduction="none")
            rgb_reference_0_loss = torch.nn.functional.cross_entropy(align_logits_reference, rgb_image_0_labels, reduction="none")
            rgb_reference_1_loss = torch.nn.functional.cross_entropy(align_logits_reference, rgb_image_1_labels, reduction="none")
            
            # we want to increase the logits of the coresponding normal & rgb images
            geo_texture_logits_per_rgb_image = model.geo_texture_logit_scale * all_rgb_image_features @ all_normal_image_features.T 
            geo_texture_logits_rgb_image_0, geo_texture_logits_rgb_image_1 = geo_texture_logits_per_rgb_image.chunk(2, dim=0)   # [batch_size, 2*batch_size]
            geo_texture_logits_per_normal_image = model.geo_texture_logit_scale * all_normal_image_features @ all_rgb_image_features.T
            geo_texture_logits_normal_image_0, geo_texture_logits_normal_image_1 = geo_texture_logits_per_normal_image.chunk(2, dim=0)  # [batch_size, 2*batch_size]

            normal_rgb_image_0_loss = torch.nn.functional.cross_entropy(geo_texture_logits_rgb_image_0, normal_image_0_labels, reduction="none")
            normal_rgb_image_1_loss = torch.nn.functional.cross_entropy(geo_texture_logits_rgb_image_1, normal_image_1_labels, reduction="none")
            rgb_normal_image_0_loss = torch.nn.functional.cross_entropy(geo_texture_logits_normal_image_0, rgb_image_0_labels, reduction="none")
            rgb_normal_image_1_loss = torch.nn.functional.cross_entropy(geo_texture_logits_normal_image_1, rgb_image_1_labels, reduction="none")

            ## dim
            label_0 = label_0.index_select(1, torch.tensor([0, 4, 3]).to(label_0.device))
            label_1 = label_1.index_select(1, torch.tensor([0, 4, 3]).to(label_1.device))

            image_0_loss = torch.stack([normal_image_0_loss, rgb_image_0_loss, rgb_normal_image_0_loss])
            image_1_loss = torch.stack([normal_image_1_loss, rgb_image_1_loss, rgb_normal_image_1_loss])
            image_0_loss, image_1_loss = image_0_loss.transpose(1,0), image_1_loss.transpose(1,0)
            eval_0_loss = torch.stack([normal_reference_0_loss, rgb_reference_0_loss, normal_rgb_image_0_loss])
            eval_1_loss = torch.stack([normal_reference_1_loss, rgb_reference_1_loss, normal_rgb_image_1_loss])
            eval_0_loss, eval_1_loss = eval_0_loss.transpose(1,0), eval_1_loss.transpose(1,0)

            contrast_loss = label_0 * image_0_loss + label_1 * image_1_loss
            eval_loss = label_0 * eval_0_loss + label_1 * eval_1_loss
        
        else:
            # geometry loss
            geo_0_logits, geo_1_logits = geo_logits_per_reference.chunk(2, dim=-1)
            index = torch.arange(geo_0_logits.shape[0], device=device, dtype=torch.long)
            geo_0_logits = geo_0_logits[index, index]
            geo_1_logits = geo_1_logits[index, index]
            geo_logits = torch.stack([geo_0_logits, geo_1_logits], dim=-1)
            geo_0_labels = torch.zeros(geo_logits.shape[0], device=device, dtype=torch.long)
            geo_1_labels = geo_0_labels + 1
            geo_0_loss = torch.nn.functional.cross_entropy(geo_logits, geo_0_labels, reduction="none")
            geo_1_loss = torch.nn.functional.cross_entropy(geo_logits, geo_1_labels, reduction="none")

            # geometry detail loss
            geo_detail_logits = torch.stack([geo_detail_logits_normal_image_0, geo_detail_logits_normal_image_1], dim=-1).squeeze(1)
            geo_detail_0_labels = torch.zeros(geo_detail_logits.shape[0], device=device, dtype=torch.int64)
            geo_detail_1_labels = geo_detail_0_labels + 1   # [batch_size]
            geo_detail_0_loss = torch.nn.functional.cross_entropy(geo_detail_logits, geo_detail_0_labels, reduction="none")
            geo_detail_1_loss = torch.nn.functional.cross_entropy(geo_detail_logits, geo_detail_1_labels, reduction="none")

            # texture loss
            texture_logits = torch.stack([texture_logits_rgb_image_0, texture_logits_rgb_image_1], dim=-1).squeeze(1)
            texture_0_labels = torch.zeros(texture_logits.shape[0], device=device, dtype=torch.int64)
            texture_1_labels = texture_0_labels + 1
            texture_0_loss = torch.nn.functional.cross_entropy(texture_logits, texture_0_labels, reduction="none")
            texture_1_loss = torch.nn.functional.cross_entropy(texture_logits, texture_1_labels, reduction="none")

            # geometry-texture loss
            index = torch.arange(geo_texture_logits_0.shape[0], device=device, dtype=torch.long)
            geo_texture_0_logits = geo_texture_logits_0[index, index]
            geo_texture_1_logits = geo_texture_logits_1[index, index]
            geo_texture_logits = torch.stack([geo_texture_0_logits, geo_texture_1_logits], dim=-1)  # (batch_size, 2)
            geo_texture_0_labels = torch.zeros(geo_texture_logits.shape[0], device=device, dtype=torch.long)
            geo_texture_1_labels = geo_texture_0_labels + 1
            geo_texture_0_loss = torch.nn.functional.cross_entropy(geo_texture_logits, geo_texture_0_labels, reduction="none")
            geo_texture_1_loss = torch.nn.functional.cross_entropy(geo_texture_logits, geo_texture_1_labels, reduction="none")

            # alignment
            align_0_logits, align_1_logits = align_logits_reference.chunk(2, dim=-1)
            index = torch.arange(align_0_logits.shape[0], device=device, dtype=torch.long)
            align_0_logits = align_0_logits[index, index]
            align_1_logits = align_1_logits[index, index]
            align_logits = torch.stack([align_0_logits, align_1_logits], dim=-1)
            align_0_labels = torch.zeros(align_logits.shape[0], device=device, dtype=torch.long)
            align_1_labels = align_0_labels + 1
            align_0_loss = torch.nn.functional.cross_entropy(align_logits, align_0_labels, reduction="none")
            align_1_loss = torch.nn.functional.cross_entropy(align_logits, align_1_labels, reduction="none")

            ## concate
            eval_0_loss = torch.stack([geo_0_loss, geo_detail_0_loss, texture_0_loss, geo_texture_0_loss, align_0_loss])
            eval_1_loss = torch.stack([geo_1_loss, geo_detail_1_loss, texture_1_loss, geo_texture_1_loss, align_1_loss])
            eval_0_loss, eval_1_loss = eval_0_loss.transpose(1,0), eval_1_loss.transpose(1,0)
            eval_loss = label_0 * eval_0_loss + label_1 * eval_1_loss

        # if we have a tie we want the logits of for each image to be equal
        # we want the ideal loss to be 0, currently, if there is a tie, it is 0.5 * log(0.5) + 0.5 * log(0.5)
        # so we add log(0.5) to the loss
        is_tie = (label_0 == label_1).float()
        is_tie *= torch.log(torch.tensor(0.5, device=device))
        eval_loss += is_tie

        # we average the image and text loss
        if self.cfg.in_batch_negatives:
            loss = (contrast_loss + eval_loss) / 2
        else:
            loss = eval_loss

        loss = loss.sum()
        return loss

    def forward(self, model, batch):
        reference_features = self.get_inference_features(
            model,
            batch[self.cfg.reference_type_column_name],
            batch[self.cfg.reference_input_column_name]
        )
        normal_image_0_features, normal_image_1_features, rgb_image_0_features, rgb_image_1_features = self.get_image_features(
            model,
            batch[self.cfg.normal_pixels_0_column_name],
            batch[self.cfg.normal_pixels_1_column_name],
            batch[self.cfg.rgb_pixels_0_column_name],
            batch[self.cfg.rgb_pixels_1_column_name]
        )

        loss = self.calc_loss(
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
        return loss
