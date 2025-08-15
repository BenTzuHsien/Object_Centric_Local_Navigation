import torch, os
import torch.nn as nn

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import grounding_dino.groundingdino.datasets.transforms as T

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_dir


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert

from pathlib import Path

SAM2_CHECKPOINT   = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG   = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

class GroundedSAM2(nn.Module):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    BOX_THRESHOLD = 0.7
    TEXT_THRESHOLD = 0.7


    def __init__(self):
        super().__init__()

        # Build Grounding DINO
        self.gdino = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT
        )

        # As we have SAM2 for SAMURAI too
        # We need to initialize Hydra and build SAM2 model
        THIS_FILE = Path(__file__).resolve()
        PROJECT_ROOT = THIS_FILE.parents[1]          # .../GroundedSAM/
        CFG_DIR = PROJECT_ROOT / "configs" / "sam2.1"  # .../GroundedSAM/configs/sam2.1
        print(CFG_DIR)

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
            sam2_model = build_sam2("sam2.1_hiera_l.yaml", SAM2_CHECKPOINT)
        GlobalHydra.instance().clear()

        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # freeze parameters
        for p in self.gdino.parameters():  
            p.requires_grad = False

        for p in self.sam2_predictor.model.parameters():  
            p.requires_grad = False

    @torch.no_grad() 
    def forward(self, image, prompt, return_mask=False, fully_masked=True):
        """
        Forward Funcion

        Parameters
        ----------
        image : PIL.Image.Image
            The current image.

        prompt : string
            prompt
        return_mask : boolean
            return_mask
        fully_masked : boolean
            fully_mask

        Returns
        -------
        outputs
            model output.
        """

        image_gdino, _ = self.transform(image, None)
        device = next(self.gdino.parameters()).device
        dtype = next(self.gdino.parameters()).dtype
        image_gdino = image_gdino.to(device=device, dtype=dtype)

        # --- Grounding‑DINO ----------------------------------------------------
        boxes, confidences, labels = predict(
            model=self.gdino,
            image=image_gdino,
            caption=prompt,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
            device=device
        )
        if boxes.numel() == 0:
            best_boxes = None
        else:
            best_boxes = boxes[confidences.argmax()]

        # --- SAM‑2 image embedding ----------------------------
        self.sam2_predictor.set_image(image)

        # SAM preprocessor upsamples input to 1024×1024.
        # Resulting token_embed shape: [B, C, H, W] = [B, 256, 64, 64], where H = W = 1024 / 16.
        token_embed = self.sam2_predictor.get_image_embedding().to(dtype=dtype)

        if best_boxes is None and fully_masked is True:
            return torch.zeros_like(token_embed), None
        
        elif best_boxes is None and fully_masked is False:
            return token_embed, None


        # .TXT FILE FOR SAMURAI AND .XYXY FOR SAM-2
        # 1. Convert box from center-based (cx, cy, w, h) in [0,1] -> pixel corners (x1, y1, x2, y2)
        w, h = image.size
        scale = torch.tensor([w, h, w, h], device=device, dtype=dtype)
        box_xyxy = box_convert(best_boxes.to(device) * scale, in_fmt="cxcywh", out_fmt="xyxy")

        # 2. Round to integer pixels
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy.tolist()]

        # 3. Turn corners into (x, y, width, height) -> for SAMURAI txt files
        x, y = x1, y1
        ww, hh = max(1, x2 - x1), max(1, y2 - y1)
        xywh_txt = (x, y, ww, hh)

        # 4. Keep original corner format -> for drawing boxes with SAM-2
        xyxy_px = (x1, y1, x2, y2)

       
        masks, _, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_xyxy,
            multimask_output=False,
        )
        mask_np = masks[0]


        # [GSAM]  raw mask [B, C, H, W] : torch.Size([1, 1, 224, 224])
        m = (torch.from_numpy(mask_np) # [H, W]
                .to(dtype=dtype, device=device)[None, None] # [B, C, H, W]
            )
        
        # [GSAM]  down-sampled mask [B, C, H, W] : torch.Size([1, 1, 64, 64])  
        m = nn.functional.interpolate(m, size=token_embed.shape[-2:], mode="nearest")
        
        # element-wise masking of the feature map, Broadcast single-channel mask across 256 channels: token_embed * m torch.Size([1, 256, 64, 64])
        feats = token_embed * m

        mask_np = masks[0] if return_mask else None

        return feats, mask_np, xywh_txt, xyxy_px
    
if __name__ == '__main__':

    from PIL import Image

    image_path = '/home/mahmu059/GroundedSAM/combined.jpg'
    image = Image.open(image_path)
    text = "green chair."
    
    gsam = GroundedSAM2()
    gsam.to("cuda")
    
    feature, mask, xywh, xyxy = gsam(image, text)

    print("xywh for SAMURAI TXT:", xywh)
    print("xyxy for SAM-2:", xyxy)


    def overlay_mask_and_box(bgr, mask, xyxy, out_path):
        vis = bgr.copy()
        if mask is not None:
            m = (mask > 0).astype(np.uint8)          
            overlay = vis.copy()
            overlay[m == 1] = (0, 255, 255)      
            vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 200, 200), 2)
        if xyxy is not None:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # bbox
        cv2.imwrite(str(out_path), vis)

    BASE_DIR   = Path("/home/mahmu059/GroundedSAM")

    with open(BASE_DIR / "first_bbox.txt", "w") as f:
        f.write("{},{},{},{}\n".format(*xywh))

    if mask is not None:
        cv2.imwrite(str(BASE_DIR / "first_mask_binary.png"), (mask.astype(np.uint8) * 255))

    overlay_mask_and_box(img_bgr, mask, xyxy, BASE_DIR / "combined_vis.jpg")
    print("Wrote:", BASE_DIR / "combined_vis.jpg", "and first_mask_binary.png")

