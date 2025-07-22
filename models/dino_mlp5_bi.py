import torch
import torch.nn as nn
from Object_Centric_Local_Navigation.models.modules.utils import resize_and_normalize_tensor
from Object_Centric_Local_Navigation.models.modules.base_model import BaseModel
from Object_Centric_Local_Navigation.models.modules.flash_cross_attention import FlashCrossAttention

class DinoMlp5Bi(BaseModel):
    """
    Inputs:
    ------
        - Current images: (B, N, C, H, W), where N = number of camera views (default 4)
        - Goal condition: Set using `set_goal(...)` before forward pass
    Outputs:
    -------
        - (B, 3, 3): Predictions over (x, y, r) targets, each with 3-class classification
    """
    TRANSFORM_SIZE = (476, 476)
    TRANSFORM_MEAN = [0.485, 0.456, 0.406]
    TRANSFORM_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super(DinoMlp5Bi, self).__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        for param in self.dinov2.parameters():
            param.requires_grad = False
        self.dinov2.eval()
        
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        num_trunk_channels = 384
        self.num_cameras = 4

        # Cross-attention block shared across cameras
        self.cross_attention = FlashCrossAttention(embed_dim=num_trunk_channels, num_heads=8)

        # Fully connected layers.
        self.fc_layer1 = nn.Sequential(
            nn.Linear(384*4*4*2, 1024),
            nn.ReLU()
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.fc_layer4 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.fc_layer_x = nn.Linear(1024, 3)
        self.fc_layer_y = nn.Linear(1024, 3)
        self.fc_layer_r = nn.Linear(1024, 3)
        self.reduce = nn.Conv2d(384, 384, kernel_size=2, stride=2)

    def set_goal(self, goal_images, text):
        """
        Set the goal condition using goal images and a language prompt.
        
        The language prompt is not used in this model.

        Parameters
        ----------
        goal_images : torch.Tensor
            A tensor of shape (N, C, H, W), N = number of cameras.
        text : str
            Text prompt describing the goal condition (Not used here).
        """
        goal_images = resize_and_normalize_tensor(goal_images, self.TRANSFORM_SIZE, self.TRANSFORM_MEAN, self.TRANSFORM_STD)
        dino_output = self.dinov2.forward_features(goal_images)
        goal_embeddings = torch.reshape(dino_output['x_norm_patchtokens'], [-1, 34, 34, 384]).permute(0, 3, 1, 2)
        self.goal_embeddings = goal_embeddings.unsqueeze(0)

    def forward(self, current_images):
        """
        Forward pass of the model.

        Parameters
        ----------
        current_images : torch.Tensor
            Shape (B, N, C, H, W), raw images.

        Returns
        -------
        outputs : torch.Tensor
            Shape (B, 3, 3): 3-class predictions over x, y, and rotation
        attention_scores : Tuple[torch.Tensor, torch.Tensor]
            A tuple of two tensors:
            - cg_attention_score : attention map from current → goal, shape depends on the attention module
            - gc_attention_score : attention map from goal → current
        """
        self.goal_embeddings = self.goal_embeddings.to(next(self.parameters()).device)
        B, N, C, H, W = current_images.shape
        current_images = current_images.reshape(B*N, C, H, W)
        current_images = resize_and_normalize_tensor(current_images, self.TRANSFORM_SIZE, self.TRANSFORM_MEAN, self.TRANSFORM_STD)

        dino_output = self.dinov2.forward_features(current_images)
        current_embeddings = torch.reshape(dino_output['x_norm_patchtokens'], [-1, 34, 34, 384]).permute(0, 3, 1, 2)
        _, C_out, H_out, W_out = current_embeddings.shape
        current_embeddings = current_embeddings.reshape(B, N, C_out, H_out, W_out)

        # Stacking 4 current features
        # Stacking 4 goal features 
        current_cat = torch.cat([current_embeddings[:, i] for i in range(self.num_cameras)], dim=3)
        goal_cat    = torch.cat([self.goal_embeddings[:, i] for i in range(self.num_cameras)], dim=3)

        current_cat = self.reduce(current_cat)  # Reduce the spatial dimensions
        goal_cat = self.reduce(goal_cat)  # Reduce the spatial dimensions
        goal_cat = goal_cat.repeat(B, 1, 1, 1)

        # Curr - Goal Cross- Attention 
        curr_goal_attenion, cg_attention_score = self.cross_attention(current_cat, goal_cat)
        curr_attended = current_cat + curr_goal_attenion

        # Goal - Current Cross- Attention
        goal_curr_attenion, gc_attention_score = self.cross_attention(goal_cat, current_cat)
        goal_attended = goal_cat + goal_curr_attenion

        # Average pooling 4x4
        curr_feat = self.global_pool(curr_attended).reshape(B, -1)
        goal_feat = self.global_pool(goal_attended).reshape(B, -1)

        # Concatenate current and goal features
        features = torch.cat([curr_feat, goal_feat], dim=1)

        # Fully connected layers.
        x = self.fc_layer1(features)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        x = self.fc_layer4(x)
        output_x = self.fc_layer_x(x)
        output_y = self.fc_layer_y(x)
        output_r = self.fc_layer_r(x)

        outputs = torch.stack([output_x, output_y, output_r], dim=1)

        return outputs, (cg_attention_score, gc_attention_score)
    
if __name__ == '__main__':

    import os
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
            transforms.Resize([640, 480]),
            transforms.ToTensor()])

    current_image_dir = ''
    goal_image_dir = ''

    current_images = []
    goal_images = []
    for i in range(4):
        current_image = Image.open(os.path.join(current_image_dir, f'{i}.jpg'))
        current_image = transform(current_image)
        current_images.append(current_image)

        goal_image = Image.open(os.path.join(goal_image_dir, f'{i}.jpg'))
        goal_image = transform(goal_image)
        goal_images.append(goal_image)

    current_images = torch.stack(current_images).to(device="cuda")
    goal_images = torch.stack(goal_images).to(device="cuda")

    model = DinoMlp5Bi().to(device="cuda")
    weight_path = ''
    model.load_weight(weight_path)
    model.set_goal(goal_images, '')

    output, _ = model(current_images.unsqueeze(0))
    print(torch.argmax(output, dim=2))