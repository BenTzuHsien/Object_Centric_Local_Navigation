import torch
import torch.nn as nn
from torchvision import transforms
from Object_Centric_Local_Navigation.models.modules.base_model import BaseModel
from Object_Centric_Local_Navigation.models.modules.flash_cross_attention import FlashCrossAttention

class DinoMlp5Uni(BaseModel):
    data_transforms = transforms.Compose([
        transforms.Resize((476, 476)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(self):
        super(DinoMlp5Uni, self).__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        for param in self.dinov2.parameters():
            param.requires_grad = False
        self.dinov2.eval()
        
        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))
        num_trunk_channels = 384
        self.num_cameras = 4

        # Cross-attention block shared across cameras
        self.cross_attention = FlashCrossAttention(embed_dim=num_trunk_channels, num_heads=8)

        # Fully connected layers.
        self.fc_layer1 = nn.Sequential(
            nn.Linear(384*8*8, 1024),
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

        #  Register a one‑byte “device sentinel” buffer
        self.register_buffer("_dev", torch.empty(0))

    def set_goal(self, goal_images, text):
        """
        Set the goal condition using goal images and a language prompt.

        Parameters
        ----------
        goal_images : list of PIL.Image.Image
            The a list of 4 PIL goal images.
        text : str
            Text prompt describing the goal condition.
        """
        device = self._dev.device
        dtype  = self._dev.dtype

        goal_embeddings = []
        for image in goal_images:
            image_tensor = self.data_transforms(image)
            image_tensor = image_tensor.to(device=device, dtype=dtype).unsqueeze(0)
            dino_output = self.dinov2.forward_features(image_tensor)
            embedding = torch.reshape(dino_output['x_norm_patchtokens'], [-1, 34, 34, 384]).squeeze(0)
            embedding = embedding.permute(2, 1, 0)
            goal_embeddings.append(embedding)

        self.goal_embeddings = torch.stack(goal_embeddings).unsqueeze(0)

    def forward(self, current_images):
        """
        Forward pass of the model.

        Parameters
        ----------
        current_images : list[list[PIL.Image.Image]]
            A batch of current observations, a 2D list (B x N) of PIL images.

        Returns
        -------
        """
        device = self._dev.device
        dtype  = self._dev.dtype
        self.goal_embeddings = self.goal_embeddings.to(device=device, dtype=dtype)

        current_embeddings = [] 
        for batch in current_images:
            batch_embeddings = []
            for image in batch:
                image_tensor = self.data_transforms(image)
                image_tensor = image_tensor.to(device=device, dtype=dtype).unsqueeze(0)
                dino_output = self.dinov2.forward_features(image_tensor)
                embedding = torch.reshape(dino_output['x_norm_patchtokens'], [-1, 34, 34, 384]).squeeze(0)
                embedding = embedding.permute(2, 1, 0)
                batch_embeddings.append(embedding)
            
            current_embeddings.append(torch.stack(batch_embeddings))
        current_embeddings = torch.stack(current_embeddings)
        
        batch_size = current_embeddings.size(0)
        # Stacking 4 current features
        # Stacking 4 goal features 
        current_cat = torch.cat([current_embeddings[:, i] for i in range(self.num_cameras)], dim=3)
        goal_cat    = torch.cat([self.goal_embeddings[:, i] for i in range(self.num_cameras)], dim=3)

        current_cat = self.reduce(current_cat)  # Reduce the spatial dimensions
        goal_cat = self.reduce(goal_cat)  # Reduce the spatial dimensions
        goal_cat = goal_cat.repeat(batch_size, 1, 1, 1)

        # Cross- Attention 
        curr_goal_attenion, attention_score = self.cross_attention(current_cat, goal_cat)
        curr_attended = current_cat + curr_goal_attenion

        # Average pooling 8x8
        features = self.global_pool(curr_attended).reshape(batch_size, -1)

        # Fully connected layers.
        x = self.fc_layer1(features)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        x = self.fc_layer4(x)
        output_x = self.fc_layer_x(x)
        output_y = self.fc_layer_y(x)
        output_r = self.fc_layer_r(x)

        outputs = torch.stack([output_x, output_y, output_r], dim=1)

        return outputs, attention_score
    
if __name__ == '__main__':

    import os
    from PIL import Image

    current_image_dir = ''
    goal_image_dir = ''

    current_images = []
    goal_images = []
    for i in range(4):
        current_image = Image.open(os.path.join(current_image_dir, f'{i}.jpg'))
        current_images.append(current_image)

        goal_image = Image.open(os.path.join(goal_image_dir, f'{i}.jpg'))
        goal_images.append(goal_image)

    model = DinoMlp5Uni().to(device="cuda", dtype=torch.float)
    weight_path = ''
    model.load_weight(weight_path)
    model.set_goal(goal_images, "green chair.")

    output, _ = model([current_images])
    print(torch.argmax(output, dim=2))