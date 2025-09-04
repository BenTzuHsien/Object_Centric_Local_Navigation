from Object_Centric_Local_Navigation.models.modules.base_model import BaseModel
from Object_Centric_Local_Navigation.models.vision_encoders.dino_v2 import DinoV2
from Object_Centric_Local_Navigation.models.segmentation_models.owl_v2_sam2 import OwlV2Sam2
from Object_Centric_Local_Navigation.models.action_decoders.decoder4 import Decoder4

class DinoDecoder4(BaseModel):

    def __init__(self, use_embeddings=False):
        
        vision_encoder = DinoV2()
        segmentation_model = OwlV2Sam2()
        action_decoder = Decoder4(vision_encoder.EMBED_DIM)
        super().__init__(vision_encoder, segmentation_model, action_decoder, use_embeddings)

if __name__ == '__main__':

    import os, torch
    from PIL import Image
    from torchvision import transforms

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

    model = DinoDecoder4().to(device='cuda')
    # weight_path = ''
    # model.load_weight(weight_path)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, current_boxes, debug_info = model(current_images.unsqueeze(0), goal_images.unsqueeze(0), prompt)
    output = torch.argmax(output, dim=2)
    print(f'Output: {output}, Box:{current_boxes}')