import torch
from typing import Optional, Tuple, List
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms

class SAM2BatchImagePredictor():

    def __init__(
            self,
            sam_model: SAM2Base,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0):
        """
        Uses SAM-2 to calculate the image embedding and mask prediction for an image given prompts.

        Parameters
        ----------
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        self.model = sam_model
        for p in self.model.parameters():  
            p.requires_grad = False
        self.model.eval()
        
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    @torch.no_grad()
    def extract_features(
            self,
            batch_images: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor]]]:
        """
        Extract image embeddings and high-resolution feature maps from a batch of images.

        This method applies preprocessing transforms, passes the images through the model's backbone,
        and returns the final embedding along with intermediate high-resolution feature maps split by batch.

        Parameters
        ----------
        batch_images : torch.Tensor
            Shape (B, 3, H, W). RGB images.

        Returns
        -------
        batch_image_embed : torch.Tensor
            Final image embeddings of shape (B, E, H', W'), where E is the embedding dimension,
            H' and W' are the spatial dimensions after backbone downsampling.
        batch_high_res_feats_split : List[Tuple[torch.Tensor, ...]]
            A list of length B (batch size), where each element is a tuple of high-resolution
            feature maps for one image. Each tensor in the tuple has shape (1, E_i, H_i, W_i),
            corresponding to intermediate stages of the backbone (excluding the final embedding).
        """
        batch_size = batch_images.shape[0]
        batch_images = self._transforms.transforms(batch_images)

        backbone_out = self.model.forward_image(batch_images)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        batch_feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        batch_image_embed = batch_feats[-1]
        batch_high_res_feats = batch_feats[:-1]
        batch_high_res_feats_split = list(zip(*[feat.split(1, dim=0) for feat in batch_high_res_feats]))

        return batch_image_embed, batch_high_res_feats_split
    
    @torch.no_grad()
    def predict_once(
            self,
            image_embed: torch.Tensor,
            high_res_feats: List[Tuple[torch.Tensor]],
            orig_hw: Tuple[int, int],
            point_coords: Optional[torch.Tensor] = None,
            point_labels: Optional[torch.Tensor] = None,
            boxes: Optional[torch.Tensor] = None,
            mask_input: Optional[torch.Tensor] = None,
            multimask_output: bool = True,
            return_logits: bool = False,
            normalize_coords: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for a single image given visual features and input prompts.

        Parameters
        ----------
        image_embed : torch.Tensor
            Image embedding of shape (1, E, H', W') from the image encoder.
            - E is the embedding channel dimension,
            - H', W' are the spatial dimensions after backbone downsampling.
        high_res_feats : Tuple[torch.Tensor, ...]
            High-resolution feature maps from earlier backbone stages.
            Each tensor has shape (1, E_i, H_i, W_i).
        orig_hw : Tuple[int, int]
            Original image height and width as (H, W) used for resizing predictions back to input scale.
        point_coords : torch.Tensor or None
            Shape (M, N, 2).  Point prompts to the model.
            M = prompt sets (objects) for the image,
            N = points per set, each point is in (X,Y) in pixels
        point_labels : torch.Tensor or None
            Shape (M, N). Labels for the point prompts.
            1 indicates a foreground point and 0 indicates a background point.
        boxes : torch.Tensor or None
            Shape (M, 4). One box per prompt set, XYXY format.
        mask_input : torch.Tensor or None
            Shape (M, 1, 256, 256). A low resolution mask input to the model, 
            typically coming from a previous prediction iteration.
            Masks returned by a previous iteration of the predict method do not need further transformation.
        multimask_output : bool, default True
            If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
        return_logits : bool, default False
            If true, returns un-thresholded masks logits instead of a binary mask.
        normalize_coords : bool, default True
            If true, the point coordinates will be normalized to the range [0,1] 
            and point_coords is expected to be wrt. image dimensions.

        Returns
        -------
        masks : torch.Tensor
            Shape (M, K, H, W), where  
            K = 1 if `multimask_output` is False,  
            K = 3 if `multimask_output` is True.  
            Each is a full-resolution binary or logit mask for the corresponding prompt.
        iou_predictions : torch.Tensor
            Shape (M, K). Containing the model's predictions for the quality of each mask.
        low_res_masks : torch.Tensor
            Shape (M, K, 256, 256). These low res logits can be passed to a subsequent iteration as mask input.
        """
        if point_coords is not None:
            point_coords = self._transforms.transform_coords(point_coords, normalize=normalize_coords, orig_hw=orig_hw)
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            boxes = self._transforms.transform_boxes(boxes, normalize=normalize_coords, orig_hw=orig_hw)  # Mx4
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # multi object prediction
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )
        # Predict masks
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_feats,
        )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, orig_hw
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks
    
    @property
    def device(self) -> torch.device:
        return self.model.device
    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype
