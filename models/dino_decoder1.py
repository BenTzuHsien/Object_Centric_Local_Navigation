from Object_Centric_Local_Navigation.models.modules.base_model import BaseModel
from Object_Centric_Local_Navigation.models.vision_encoders.dino_v2 import DinoV2
from Object_Centric_Local_Navigation.models.segmentation_models.owl_v2_sam2 import OwlV2Sam2
from Object_Centric_Local_Navigation.models.action_decoders.decoder1 import Decoder1

class DinoDecoder1(BaseModel):

    def __init__(self, use_embeddings=False):
        
        vision_encoder = DinoV2()
        segmentation_model = OwlV2Sam2()
        action_decoder = Decoder1()
        super().__init__(vision_encoder, segmentation_model, action_decoder, use_embeddings)

if __name__ == '__main__':

    import os, torch
    from PIL import Image
    from torchvision import transforms
    from torchvision.utils import save_image

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

    model = DinoDecoder1().to(device='cuda')
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