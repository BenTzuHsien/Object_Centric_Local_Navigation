import torch
import torch.nn as nn
from Object_Centric_Local_Navigation.models.modules.base_model import BaseModel
from Object_Centric_Local_Navigation.models.backbones.grounded_sam2 import GroundedSAM2
from Object_Centric_Local_Navigation.models.modules.flash_cross_attention import FlashCrossAttention

class GsamMlp5Bi(BaseModel):
    """
    Inputs:
    ------
        - Current images: (B, N, C, H, W), where N = number of camera views (default 4)
        - Goal condition: Set using `set_goal(...)` before forward pass
    Outputs:
    -------
        - (B, 3, 3): Predictions over (x, y, r) targets, each with 3-class classification
    """
    def __init__(self, use_gsam=True):
        super().__init__()

        if use_gsam:
            self.use_gsam = True
            self.gsam = GroundedSAM2()
        else:
            self.use_gsam = False

        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        num_trunk_channels = 256
        self.num_cameras = 4 

        self.cross_attention = FlashCrossAttention(embed_dim=num_trunk_channels, num_heads=8)
        
        # Fully connected layers.
        self.fc_layer1 = nn.Sequential(
            nn.Linear(256*4*4*2, 1024),
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
        self.reduce = nn.Conv2d(256, 256, kernel_size=2, stride=2)

    def set_goal(self, goal_images, text):
        """
        Set the goal condition using goal images and a language prompt.
        
        If `use_gsam` is False, pass in the embeddings extracted by grounded SAM2 model instead of raw images.

        Parameters
        ----------
        goal_images : torch.Tensor
            If `use_gsam` is True: a tensor of shape (N, C, H, W), N = number of cameras.
            If `use_gsam` is False: a tensor of shape (1, N, C', H', W'), precomputed embeddings.
        text : str
            Text prompt describing the goal condition (used by GSAM).
        """
        if self.use_gsam:
            prompts = [text] * self.num_cameras
            goal_embeddings, _ = self.gsam(goal_images, prompts)
            self.goal_embeddings = goal_embeddings.unsqueeze(0)
        else:
            self.goal_embeddings = goal_images

        self.prompt = text

    def forward(self, current_images):
        """
        Forward pass of the model.

        Parameters
        ----------
        current_images : torch.Tensor
            If `use_gsam` is True: shape (B, N, C, H, W), raw images
            If `use_gsam` is False: shape (B, N, C', H', W'), precomputed embeddings

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

        if self.use_gsam:
            current_images = current_images.reshape(B*N, C, H, W)
            prompts = [self.prompt] * (B * N)
            current_embeddings, _ = self.gsam(current_images, prompts)
            _, C_out, H_out, W_out = current_embeddings.shape
            current_embeddings = current_embeddings.reshape(B, N, C_out, H_out, W_out)
        else:
            current_embeddings = current_images

        # Features from GSAM
        # [GSAM-MLP] curr torch.Size([64, 4, 256, 64, 64])  goal torch.Size([64, 4, 256, 64, 64])
        # print(f"[GSAM-MLP] curr {current_embeddings.shape}  goal {self.goal_embeddings.shape}")

        # Stacking 4 current features
        # Stacking 4 goal features 
        current_cat = torch.cat([current_embeddings[:, i] for i in range(self.num_cameras)], dim=3)
        goal_cat    = torch.cat([self.goal_embeddings[:, i] for i in range(self.num_cameras)], dim=3)
        # [GSAM-MLP] current_cat torch.Size([64, 256, 64, 256])  goal_cat torch.Size([64, 256, 64, 256])
        # print(f"[GSAM-MLP] current_cat {current_cat.shape}  goal_cat {goal_cat.shape}")

        current_cat = self.reduce(current_cat)  # Reduce the spatial dimensions
        goal_cat = self.reduce(goal_cat)  # Reduce the spatial dimensions
        goal_cat = goal_cat.repeat(B, 1, 1, 1)

        # Curr - Goal Cross- Attention 
        curr_goal_attenion, cg_attention_score = self.cross_attention(current_cat, goal_cat)
        curr_attended = current_cat + curr_goal_attenion
        # [GSAM-MLP] curr_att torch.Size([64, 256, 32, 128])
        # print(f"[GSAM-MLP] curr_att {curr_attended.shape}")

        # Goal - Current Cross- Attention
        goal_curr_attenion, gc_attention_score = self.cross_attention(goal_cat, current_cat)
        goal_attended = goal_cat + goal_curr_attenion
        # [GSAM-MLP] goal_att torch.Size([64, 256, 32, 128]
        # print(f"[GSAM-MLP] goal_att {goal_attended.shape}")

        # Average pooling 4x4
        curr_feat = self.global_pool(curr_attended).reshape(B, -1)  
        goal_feat = self.global_pool(goal_attended).reshape(B, -1)     
        # [GSAM-MLP] curr_pool torch.Size([64, 4096])  goal_pool torch.Size([64, 4096])  
        # print(f"[GSAM-MLP] curr_pool {curr_feat.shape}  goal_pool {goal_feat.shape}")
        
        # Concatenate current and goal features
        features = torch.cat([curr_feat, goal_feat], dim=1)
        # [GSAM-MLP] concat features torch.Size([64, 8192]
        # print(f"[GSAM-MLP] concat features {features.shape}")

        # MLP Layers
        x = self.fc_layer1(features)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        x = self.fc_layer4(x)
        # [GSAM-MLP] after FC4 torch.Size([64, 1024])
        # print(f"[GSAM-MLP] after FC4 {x.shape}")

        output_x = self.fc_layer_x(x)
        output_y = self.fc_layer_y(x)
        output_r = self.fc_layer_r(x)

        outputs = torch.stack([output_x, output_y, output_r], dim=1)
        # [GSAM-MLP] outputs torch.Size([64, 3, 3])
        # print(f"[GSAM-MLP] outputs {outputs.shape}")

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

    model = GsamMlp5Bi().to(device="cuda")
    weight_path = ''
    model.load_weight(weight_path)
    model.set_goal(goal_images, '')

    output, _ = model(current_images.unsqueeze(0))
    print(torch.argmax(output, dim=2))