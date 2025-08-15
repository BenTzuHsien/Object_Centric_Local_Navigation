import torch
from collections import OrderedDict

class BaseModel(torch.nn.Module):

    def __init__(self, vision_encoder, segmentation_model, action_decoder, use_embeddings=False):
        super().__init__()

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

    @torch.no_grad()
    def set_goal(self, goal_images, furniture_prompt):
        """
        Set the goal condition using goal images or precomputed goal embeddings.

        If `use_embeddings` is False, the goal images will be encoded and segmented
        to produce the goal embeddings.

        Parameters
        ----------
        goal_images : torch.Tensor
            If `use_embeddings` is False:
                Shape (N, C, H, W), raw RGB goal images.
            If `use_embeddings` is True:
                Shape (1, C', H', N*W'), precomputed goal embeddings for four images
                concatenated along the width dimension.
        furniture_prompt : str
            Text prompt describing the target furniture (used by the segmentation model).

        Returns
        -------
        goal_masks : List[Optional[torch.Tensor]]
            List of length N; each element is either None or a mask tensor (1, H, W).
        """
        self.prompt = furniture_prompt
        if not self.use_embeddings:
            goal_embeds = self.vision_encoder(goal_images)

            goal_embeddings, goal_masks = self.segmentation_model(goal_images, [furniture_prompt] * goal_images.shape[0], goal_embeds)
            _, C_out, H_out, W_out = goal_embeddings.shape
            self.goal_embeddings = goal_embeddings.permute(1, 2, 0, 3).reshape(C_out, H_out, goal_images.shape[0] * W_out)   # Concatenate the images horizontally
            return goal_masks

        else:
            self.goal_embeddings = goal_images
            return None

    def forward(self, current_images):
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
        self.goal_embeddings = self.goal_embeddings.to(next(self.parameters()).device)

        masks = None
        if not self.use_embeddings:
            B, N, C, H, W = current_images.shape
            current_images = current_images.reshape(B*N, C, H, W)
            prompts = [self.prompt] * (B * N)
            current_embeds = self.vision_encoder(current_images)

            current_embeddings, masks = self.segmentation_model(current_images, prompts, current_embeds)
            _, C_out, H_out, W_out = current_embeddings.shape
            current_embeddings = current_embeddings.reshape(B, N, C_out, H_out, W_out)
            current_embeddings = current_embeddings.permute(0, 2, 3, 1, 4).reshape(B, C_out, H_out, N * W_out)   # Concatenate the images horizontally
        
        else:
            B, C, H, W = current_images.shape
            current_embeddings = current_images
        
        goal_embeddings = self.goal_embeddings.expand(B, -1, -1, -1)
        action, decoder_debug_info = self.action_decoder(current_embeddings, goal_embeddings)

        return action, (masks, decoder_debug_info)

if __name__ == '__main__':

    import os
    from PIL import Image
    from torchvision import transforms
    from torchvision.utils import save_image

    from Object_Centric_Local_Navigation.models.vision_encoders.dino_v2 import DinoV2
    from Object_Centric_Local_Navigation.models.segmentation_models.owl_v2_sam2 import OwlV2Sam2
    from Object_Centric_Local_Navigation.models.action_decoders.decoder1 import Decoder1

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])

    goal_images_dir = ''
    current_image_dir = ''

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
    goal_masks = model.set_goal(goal_images, '')

    # Mask Visualization
    masked_goal_images = []
    for i in range(4):
        if goal_masks[i] is not None:
            masked_image = goal_images[i] * goal_masks[i]
        else:
            masked_image = torch.zeros_like(goal_images[i])
        masked_goal_images.append(masked_image)
    masked_goal_images = torch.cat(masked_goal_images, dim=2)
    save_image(masked_goal_images, 'masked_goal_image.jpg')

    output, debug_info = model(current_images.unsqueeze(0))
    print(torch.argmax(output, dim=2))
    
    # Mask Visualization
    masks = debug_info[0]
    masked_images = []
    for i in range(4):
        if masks[i] is not None:
            masked_image = current_images[i] * masks[i]
        else:
            masked_image = torch.zeros_like(current_images[i])
        
        masked_images.append(masked_image)
    masked_images = torch.cat(masked_images, dim=2)
    save_image(masked_images, 'masked_current_image.jpg')