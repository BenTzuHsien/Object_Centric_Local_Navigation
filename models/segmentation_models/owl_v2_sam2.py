import torch, os
from transformers import Owlv2Processor, Owlv2ForObjectDetection, Owlv2ImageProcessorFast, CLIPTokenizer
from sam2.build_sam import build_sam2
from Object_Centric_Local_Navigation.models.modules.sam2_batch_image_predictor import SAM2BatchImagePredictor

class OwlV2Sam2(torch.nn.Module):
    MODEL_NAME = "google/owlv2-base-patch16-ensemble"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    SAM2_CHECKPOINT = os.path.expanduser("/opt/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    THRESHOLD = 0.2

    def __init__(self):
        super().__init__()
        image_processor_fast = Owlv2ImageProcessorFast.from_pretrained(self.MODEL_NAME)
        tokenizer = CLIPTokenizer.from_pretrained(self.MODEL_NAME)
        self.processor = Owlv2Processor(image_processor=image_processor_fast, tokenizer=tokenizer)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.MODEL_NAME)
        for p in self.model.parameters():  
            p.requires_grad = False
        self.model.eval()

        # Build SAM-2
        self.sam2_model = build_sam2(self.SAM2_MODEL_CONFIG, self.SAM2_CHECKPOINT)
        self.sam2_predictor = SAM2BatchImagePredictor(self.sam2_model)

    @torch.no_grad() 
    def forward(self, batch_images, prompts, batch_embeddings):

        batch_size, _, H, W = batch_images.shape

        inputs = self.processor(text=prompts, images=batch_images, return_tensors="pt", do_rescale=False).to(self.device)
        outputs = self.model(**inputs)

        # Convert output boxes
        target_sizes = torch.tensor([(H, W)] * batch_size)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.THRESHOLD, text_labels=prompts
        )

        # Extract SAM2 Embeddings
        batch_image_embed, batch_high_res_feats_split = self.sam2_predictor.extract_features(batch_images)

        # results are a list of length batch_size. In each result, the box are shape of (M, 4); M is the number of boxes.

        masked_embeddings = []
        masks = []
        for i in range(batch_size):
            if results[i]['boxes'].shape[0] == 0:
                masked_embeddings.append(torch.zeros_like(batch_embeddings[i]))
                masks.append(None)
            else:
                best_box = results[i]['boxes'][results[i]['scores'].argmax()]

                image_mask, _, _ = self.sam2_predictor.predict_once(
                    batch_image_embed[i].unsqueeze(0), 
                    batch_high_res_feats_split[i],
                    (H, W),
                    boxes=best_box.unsqueeze(0), 
                    multimask_output=False)
                image_mask = image_mask.float()

                embedding_mask = torch.nn.functional.interpolate(image_mask, batch_embeddings.shape[-2:], mode="nearest")
                masked_embed = embedding_mask * batch_embeddings[i]
                
                masked_embeddings.append(masked_embed.squeeze(0))
                masks.append(image_mask.squeeze(0))

        masked_embeddings = torch.stack(masked_embeddings)
        # return results, None
        return masked_embeddings, masks
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

if __name__ == '__main__':

    import os
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
    
    prompts = [['']]* 4
    
    segmentation_model = OwlV2Sam2()
    segmentation_model.cuda()
    
    images = images.to(segmentation_model.device)
    masked_embeddings, masks = segmentation_model(images, prompts, torch.rand([4, 256, 64, 64]).cuda())
    
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