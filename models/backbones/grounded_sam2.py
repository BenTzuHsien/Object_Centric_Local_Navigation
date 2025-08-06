import torch, os
import torch.nn as nn

from grounding_dino.groundingdino.util.inference import load_model
from sam2.build_sam import build_sam2
from torchvision.ops import box_convert

from Object_Centric_Local_Navigation.models.modules.gdino_batch_image_predictor import GDinoBatchImagePredictor
from Object_Centric_Local_Navigation.models.modules.sam2_batch_image_predictor import SAM2BatchImagePredictor

GROUNDING_DINO_CONFIG = os.path.expanduser("/opt/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = os.path.expanduser("/opt/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth")
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = os.path.expanduser("/opt/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")

class GroundedSAM2(nn.Module):
    BOX_THRESHOLD = 0.45
    TEXT_THRESHOLD = 0.4

    def __init__(self):
        super().__init__()

        # Build Grounding DINO
        self.gdino_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT
        )
        self.gdino_predictor = GDinoBatchImagePredictor(self.gdino_model)

        # Build SAM-2
        self.sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT)
        self.sam2_predictor = SAM2BatchImagePredictor(self.sam2_model)

    @torch.no_grad() 
    def forward(self, images, prompts, return_mask=False):
        """
        Compute masked SAM-2 image embeddings based on GroundingDINO-predicted boxes.

        This method extracts image features using SAM-2 and selects the best object
        from GroundingDINO predictions per image. If a valid box is found, the SAM-2
        feature map is masked accordingly; otherwise, a zero tensor is returned.

        Parameters
        ----------
        images : torch.Tensor
            Input batch of RGB images of shape (B, 3, H, W).
        prompts : List[str]
            A list of text prompts corresponding to the input images.
        return_mask : bool, default False
            Whether to return the predicted binary masks along with the features.

        Returns
        -------
        feature_maps : torch.Tensor
            A batched tensor of masked SAM-2 embeddings of shape (B, C, H', W'),
            where C is the feature dimension (256), and H', W' are spatial dimensions (64, 64).
            If no object is detected for an image, a zero tensor is returned for that sample.
        masks : List[torch.Tensor or None] or None
            A list of binary masks of shape (1, H, W) for each image if `return_mask=True`.
            The list contains `None` for samples with no valid detections.
            If `return_mask=False`, returns `None`.
        """
        batch_size, _, H, W = images.shape

        # Extract SAM2 Embeddings
        batch_image_embed, batch_high_res_feats_split = self.sam2_predictor.extract_features(images)
        
        # Grounding‑DINO
        boxes_list, confidences_list, labels_list = self.gdino_predictor.predict(images, prompts, self.BOX_THRESHOLD, self.TEXT_THRESHOLD)
        
        feature_maps = []
        masks = []
        scale = torch.tensor([W, H, W, H], device=self.device, dtype=self.dtype)
        for i in range(batch_size):
            if boxes_list[i].numel() == 0:
                feature_maps.append(torch.zeros_like(batch_image_embed[i]))
                masks.append(None)
            else:
                best_box = boxes_list[i][confidences_list[i].argmax()]
                box_xyxy = box_convert(best_box * scale, in_fmt="cxcywh", out_fmt="xyxy")
                
                image_mask, _, _ = self.sam2_predictor.predict_once(
                    batch_image_embed[i].unsqueeze(0), 
                    batch_high_res_feats_split[i],
                    (H, W),
                    boxes=box_xyxy.unsqueeze(0), 
                    multimask_output=False)
                
                feature_mask = image_mask.float()
                feature_mask = nn.functional.interpolate(feature_mask, batch_image_embed[i].shape[-2:], mode="nearest").squeeze(0)
                masked_feature = feature_mask * batch_image_embed[i]
                feature_maps.append(masked_feature)
                masks.append(image_mask.squeeze(0))

        feature_maps = torch.stack(feature_maps)
        return (feature_maps, masks) if return_mask else (feature_maps, None)
    
    @property
    def device(self) -> torch.device:
        return self.gdino_predictor.device
    @property
    def dtype(self) -> torch.dtype:
        return self.gdino_predictor.dtype
    
if __name__ == '__main__':

    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])

    image_dir = ''
    images = []
    for i in range(4):
        image_path = os.path.join(image_dir, f'{i}.jpg')
        image = Image.open(image_path)
        image_tensor = transform(image)
        images.append(image_tensor)

    images = torch.stack(images)
    print(images.shape)
    
    prompts = ['green chair.'] * 4
    
    gsam = GroundedSAM2()
    gsam.to("cuda")
    
    images = images.to(gsam.device)
    features, _ = gsam(images, prompts)
    print(f'feature shape: {features.shape}')
    
    for i, feature in enumerate(features):
        magnitude = torch.norm(feature.permute(1, 2, 0), dim=-1)

        from torchvision.transforms.functional import to_pil_image
        min_val = magnitude.min()
        max_val = magnitude.max()
        feature_vis = (((magnitude - min_val) / (max_val - min_val + 1e-8)) * 225).to(torch.uint8)
        img = to_pil_image(feature_vis)
        img.save(f"magnitude_map_{i}.png")