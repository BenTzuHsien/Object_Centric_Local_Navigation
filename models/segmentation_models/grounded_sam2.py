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
    def forward(self, batch_images, prompts, batch_embeddings):
        """
        Apply GroundedSAM2 to precomputed image embeddings.

        For each image, GroundingDINO predicts candidate boxes from the text prompt.
        The highest-confidence box is chosen and converted from normalized (cx, cy, w, h)
        to pixel (x1, y1, x2, y2) coordinates. SAM-2 then predicts a binary mask for that
        box; the mask is resized to the spatial size of the provided embeddings and used
        to zero-out features outside the object. If no valid box is found, a zero tensor
        is returned for that sample and the mask is `None`.

        Parameters
        ----------
        images : torch.Tensor
            Input batch of RGB images of shape (B, 3, H, W).
        prompts : List[str]
            A list of text prompts corresponding to the input images.
        batch_embeddings : torch.Tensor
            Precomputed image embeddings to be masked, of shape (B, C, H', W').

        Returns
        -------
        masked_embeddings : torch.Tensor
            Masked embeddings of shape (B, C, H', W'). For images without a valid
            detection, the corresponding slice is all zeros.
        masks : List[torch.Tensor or None]
            Per-image binary masks of shape (1, H, W). 
            Entries are `None` when no valid detection exists for that image.
        """
        batch_size, _, H, W = batch_images.shape

        # Extract SAM2 Embeddings
        batch_image_embed, batch_high_res_feats_split = self.sam2_predictor.extract_features(batch_images)
        
        # Grounding‑DINO
        boxes_list, confidences_list, labels_list = self.gdino_predictor.predict(batch_images, prompts, self.BOX_THRESHOLD, self.TEXT_THRESHOLD)
        
        masked_embeddings = []
        masks = []
        scale = torch.tensor([W, H, W, H], device=self.device, dtype=self.dtype)
        for i in range(batch_size):
            if boxes_list[i].numel() == 0:
                masked_embeddings.append(torch.zeros_like(batch_embeddings[i]))
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
                image_mask = image_mask.float()
                
                embedding_mask = torch.nn.functional.interpolate(image_mask, batch_embeddings.shape[-2:], mode="nearest")
                masked_embed = embedding_mask * batch_embeddings[i]
                
                masked_embeddings.append(masked_embed.squeeze(0))
                masks.append(image_mask.squeeze(0))
        
        masked_embeddings = torch.stack(masked_embeddings)
        return masked_embeddings, masks
    
    @property
    def device(self) -> torch.device:
        return self.gdino_predictor.device
    @property
    def dtype(self) -> torch.dtype:
        return self.gdino_predictor.dtype
    
if __name__ == '__main__':

    from PIL import Image
    from torchvision import transforms
    from torchvision.utils import save_image

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])

    images_dir = ''
    images = []
    for i in range(4):
        image_path = os.path.join(images_dir, f'{i}.jpg')
        image = Image.open(image_path)
        image_tensor = transform(image)
        images.append(image_tensor)

    images = torch.stack(images)
    print(images.shape)
    
    prompts = [''] * 4
    
    gsam = GroundedSAM2()
    gsam.cuda()
    
    images = images.to(gsam.device)
    masked_embeddings, masks = gsam(images, prompts, torch.rand([4, 256, 64, 64]).cuda())
    
    masked_images = []
    magnitude_images = []
    for i in range(4):

        if masks[i] is not None:
            masked_image = images[i] * masks[i]
            magnitude = torch.norm(masked_embeddings[i].permute(1, 2, 0), dim=-1)
        else:
            masked_image = torch.zeros_like(images[i])
            magnitude = torch.zeros([64, 64]).cuda()
        
        masked_images.append(masked_image)
        magnitude_images.append(magnitude)

    masked_images = torch.cat(masked_images, dim=2)
    magnitude_images = torch.cat(magnitude_images, dim=1)
    save_image(masked_images, 'masked_image.jpg')
    save_image(magnitude_images, 'magnitude_image.jpg')
