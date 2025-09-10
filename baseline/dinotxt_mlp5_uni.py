import torch
from collections import OrderedDict
from dinov2.hub.dinotxt import get_tokenizer
from Object_Centric_Local_Navigation.models.vision_encoders.dino_v2 import DinoV2

class DinotxtMlp5Uni(torch.nn.Module):
    TRANSFORM_SIZE = (476, 476)
    TRANSFORM_MEAN = [0.485, 0.456, 0.406]
    TRANSFORM_STD = [0.229, 0.224, 0.225]
    EMBED_DIM = 1024

    def __init__(self, use_embeddings=False):
        super().__init__()
        
        self.use_embeddings = use_embeddings
        if not use_embeddings:
            self.dinov2 = DinoV2()
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()

            self.dinotxt = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l')
            for param in self.dinotxt.parameters():
                param.requires_grad = False
            self.dinotxt.eval()

            self.avg_pool = torch.nn.AdaptiveAvgPool2d((16, 16 * 4))

        self.tokenizer = get_tokenizer()
        self.attention_current_txt = torch.nn.MultiheadAttention(embed_dim=self.EMBED_DIM, num_heads=8, batch_first=True)
        self.attention_goal_txt = torch.nn.MultiheadAttention(embed_dim=self.EMBED_DIM, num_heads=8, batch_first=True)
        self.attention = torch.nn.MultiheadAttention(embed_dim=self.EMBED_DIM, num_heads=1, batch_first=True)
        
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(self.EMBED_DIM, 1024),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(1024),
            torch.nn.Linear(1024, 1024),
            torch.nn.SiLU()
        )
        self.fc_layer_x = torch.nn.Linear(1024, 3)
        self.fc_layer_y = torch.nn.Linear(1024, 3)
        self.fc_layer_r = torch.nn.Linear(1024, 3)

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

    def forward(self, current_images, goal_images, furniture_prompt):

        if not self.use_embeddings:

            B, N, C, H, W = current_images.shape

            # Process goal images
            goal_images = goal_images.reshape(B*N, C, H, W)
            goal_features = self.dinov2(goal_images)

            _, C_out, H_out, W_out = goal_features.shape
            goal_features = goal_features.reshape(B, N, C_out, H_out, W_out).permute(0, 2, 3, 1, 4).reshape(B, C_out, H_out, N * W_out)

            ## Pooling
            goal_features = self.avg_pool(goal_features)
            goal_features = goal_features.flatten(start_dim=2).permute(0, 2, 1)

            # ----- Output: goal_features -----

            # Process current images
            current_images = current_images.reshape(B*N, C, H, W)
            current_features = self.dinov2(current_images)
            
            _, C_out, H_out, W_out = current_features.shape
            current_features = current_features.reshape(B, N, C_out, H_out, W_out).permute(0, 2, 3, 1, 4).reshape(B, C_out, H_out, N * W_out)

            ## Pooling
            current_features = self.avg_pool(current_features)
            current_features = current_features.flatten(start_dim=2).permute(0, 2, 1)

            # ----- Output: current_features -----

        else:
            B = current_images.shape[0]
            current_features = current_images
            goal_features = goal_images

        # Process Text
        furniture_prompt = self.tokenizer.tokenize([furniture_prompt] * B).to(current_images.device)
        prompt_features = self.dinotxt.encode_text(furniture_prompt)[:, 1024:]
        prompt_features = prompt_features.unsqueeze(1)

        # ----- Output: prompt_features -----

        # Action Decoder
        current_attend, _ = self.attention_current_txt(prompt_features, current_features, current_features)
        goal_attend, _ = self.attention_goal_txt(prompt_features, goal_features, goal_features)
        features, _ = self.attention(current_attend, goal_attend, goal_attend)

        features = features.reshape(B, -1)
        hidden = self.fc_layer(features)

        output_x = self.fc_layer_x(hidden)
        output_y = self.fc_layer_y(hidden)
        output_r = self.fc_layer_r(hidden)

        outputs = torch.stack([output_x, output_y, output_r], dim=1)
        return outputs, None, None
    
if __name__ == '__main__':

    import os
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])

    goal_images_dir = '/data/SPOT_Real_World_Dataset/green_chair/Goal_Images'
    current_image_dir = '/data/SPOT_Real_World_Dataset/green_chair/000/00'
    prompt = 'green chair.'

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

    model = DinotxtMlp5Uni().to(device='cuda')
    # weight_path = ''
    # model.load_weight(weight_path)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, current_boxes, debug_info = model(current_images.unsqueeze(0), goal_images.unsqueeze(0), prompt)
    output = torch.argmax(output, dim=2)
    print(f'Output: {output}, Box:{current_boxes}')