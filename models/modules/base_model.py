import torch
from collections import OrderedDict
from Object_Centric_Local_Navigation.models.modules.utils import get_masked_region

class BaseModel(torch.nn.Module):

    def __init__(self, vision_encoder, segmentation_model, action_decoder, use_embeddings=False):
        super().__init__()

        self.avg_pool = torch.nn.AdaptiveAvgPool2d((16, 16))

        self.use_embeddings = use_embeddings
        if not use_embeddings:
            self.vision_encoder = vision_encoder
            self.segmentation_model = segmentation_model
        
        self.action_decoder = action_decoder

    def load_weights(self, weight_path):
        """
        Load model weights from a file, with support for DataParallel-trained checkpoints.

        Parameters
        ----------
        weight_path : str
            Path to the weight (.pth) file.
        """
        state_dict = torch.load(weight_path, map_location=next(self.parameters()).device)
        if any(k.startswith("module.") for k in state_dict.keys()):
            # Trained with DataParallel, strip "module."
            new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
        else:
            new_state_dict = state_dict
        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, current_images, goal_images, furniture_prompt=None, previous_bounding_box=None):
        """
        Forward pass of the model.

        Parameters
        ----------
        current_images : torch.Tensor
            If `use_embeddings` is False:
                Shape (B, N, C, H, W), where N is the number of camera views.
                Raw RGB images from multiple views.
            If `use_embeddings` is True:
                Shape (B, C', H', N*W'), precomputed embeddings for four views
                concatenated along width.

        Returns
        -------
        action : torch.Tensor
            3-class predictions over x, y, and rotation. Shape (B, 3, 3).
        debug_info : Tuple[List[Optional[torch.Tensor]], torch.Tensor]
            A tuple containing:
            - masks : List[Optional[torch.Tensor]]
                List of length B*N; each element is either None or a mask tensor (1, H, W).
            - score_matrix : torch.Tensor
                Score matrix between current and goal embeddings.
                Shape (B, H' * N*W', H' * N*W')
        """
        current_masks, goal_masks = None, None
        if not self.use_embeddings:
            B, N, C, H, W = current_images.shape
            prompts = [furniture_prompt] * B
        
            # Process goal images
            goal_panoramic = goal_images.permute(0, 2, 3, 1, 4).reshape(B, C, H, N * W)
            
            ## Encode goal images
            goal_images = goal_images.reshape(B*N, C, H, W)
            goal_embeds = self.vision_encoder(goal_images)

            _, C_out, H_out, W_out = goal_embeds.shape
            goal_embeds = goal_embeds.reshape(B, N, C_out, H_out, W_out)
            goal_embeds = goal_embeds.permute(0, 2, 3, 1, 4).reshape(B, C_out, H_out, N * W_out)

            ## Segment goal imgaes
            goal_boxes, goal_masks = self.segmentation_model(goal_panoramic, prompts, previous_bounding_box)

            embedding_masks = torch.nn.functional.interpolate(goal_masks, [H_out, N * W_out], mode="nearest")
            embedding_boxes = get_masked_region(embedding_masks)

            goal_embeddings = []
            for i in range(B):
                x1, y1, x2, y2 = embedding_boxes[i]
                masked_embeddings = embedding_masks[i][:, y1:y2+1, x1:x2+1] * goal_embeds[i][:, y1:y2+1, x1:x2+1]
                masked_embeddings = self.avg_pool(masked_embeddings)
                goal_embeddings.append(masked_embeddings)
            goal_embeddings = torch.stack(goal_embeddings)

            # ----- Output: goal_boxes, goal_embeddings -----

            # Process current images
            current_panoramic = current_images.permute(0, 2, 3, 1, 4).reshape(B, C, H, N * W)

            ## Encode current images
            current_images = current_images.reshape(B*N, C, H, W)
            current_embeds = self.vision_encoder(current_images)

            _, C_out, H_out, W_out = current_embeds.shape
            current_embeds = current_embeds.reshape(B, N, C_out, H_out, W_out)
            current_embeds = current_embeds.permute(0, 2, 3, 1, 4).reshape(B, C_out, H_out, N * W_out)

            ## Segment current images
            current_boxes, current_masks = self.segmentation_model(current_panoramic, prompts, previous_bounding_box)

            embedding_masks = torch.nn.functional.interpolate(current_masks, [H_out, N * W_out], mode="nearest")
            embedding_boxes = get_masked_region(embedding_masks)

            current_embeddings = []
            for i in range(B):
                x1, y1, x2, y2 = embedding_boxes[i]
                masked_embeddings = embedding_masks[i][:, y1:y2+1, x1:x2+1] * current_embeds[i][:, y1:y2+1, x1:x2+1]
                masked_embeddings = self.avg_pool(masked_embeddings)
                current_embeddings.append(masked_embeddings)
            current_embeddings = torch.stack(current_embeddings)

            # ----- Output: current_boxes, current_embeddings
        
        else:
            current_boxes = current_images[0]
            current_embeddings = current_images[1]
            goal_boxes = goal_images[0]
            goal_embeddings = goal_images[1]
        
        action, decoder_debug_info = self.action_decoder(current_boxes, current_embeddings, goal_boxes, goal_embeddings)

        return action, ((current_masks, goal_masks), decoder_debug_info)

if __name__ == '__main__':

    import os
    from PIL import Image
    from torchvision import transforms

    from Object_Centric_Local_Navigation.models.vision_encoders.dino_v2 import DinoV2
    from Object_Centric_Local_Navigation.models.segmentation_models.owl_v2_sam2 import OwlV2Sam2
    from Object_Centric_Local_Navigation.models.action_decoders.decoder1 import Decoder1

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])

    goal_images_dir = ''
    current_image_dir = ''
    prompt = ''

    goal_images = []
    current_images = []
    for i in range(4):
        current_image = Image.open(os.path.join(current_image_dir, f'{i}.jpg'))
        current_image = transform(current_image)
        current_images.append(current_image)

        goal_image = Image.open(os.path.join(goal_images_dir, f'{i}.jpg'))
        goal_image = transform(goal_image)
        goal_images.append(goal_image)
    current_images = torch.stack(current_images).to(device='cuda')
    goal_images = torch.stack(goal_images).to(device='cuda')

    vision_encoder=DinoV2()
    segmentation_model=OwlV2Sam2()
    action_decoder=Decoder1()
    model = BaseModel(vision_encoder, segmentation_model, action_decoder).to(device='cuda')
    # weight_path = ''
    # model.load_weight(weight_path)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, debug_info = model(current_images.unsqueeze(0), goal_images.unsqueeze(0), prompt)
    output = torch.argmax(output, dim=2)
    print(f'Output: {output}')
