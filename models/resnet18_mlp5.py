import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from Object_Centric_Local_Navigation.models.modules.base_model import BaseModel
from Object_Centric_Local_Navigation.models.modules.flash_cross_attention import FlashCrossAttention

class Resnet18Mlp5(BaseModel):
    data_transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def __init__(self):
        super(Resnet18Mlp5, self).__init__()
        # Shared ResNet18 trunk (excluding the last 2 layers)
        base_resnet = resnet18(weights='DEFAULT')
        self.resnet18 = nn.Sequential(*list(base_resnet.children())[:-2])
        
        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))
        num_trunk_channels = 512
        self.num_cameras = 4

        # Cross-attention block shared across cameras
        self.cross_attention = FlashCrossAttention(embed_dim=num_trunk_channels, num_heads=8)

        # Fully connected layers.
        # Input feature dimension: 5 cameras * 2 (current + goal) * 512 = 5120.
        self.fc_layer1 = nn.Sequential(
            nn.Linear(512*8*8, 1024),
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
        device = next(self.fc_layer1.parameters()).device
        dtype = next(self.fc_layer1.parameters()).dtype

        goal_image_tensors = []
        for image in goal_images:
            image_tensor = self.data_transforms(image)
            image_tensor = image_tensor.to(device=device, dtype=dtype)
            goal_image_tensors.append(image_tensor)

        self.goal_images = torch.stack(goal_image_tensors)

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
        device = next(self.fc_layer1.parameters()).device
        dtype = next(self.fc_layer1.parameters()).dtype
        self.goal_images.to(device=device, dtype=dtype)

        # Process goal images through Resnet18
        goal_embeddings = self.resnet18(self.goal_images).unsqueeze(0)

        current_embeddings = [] 
        for batch in current_images:
            batch_embeddings = []
            for image in batch:
                image_tensor = self.data_transforms(image)
                image_tensor = image_tensor.to(device=device, dtype=dtype).unsqueeze(0)
                embedding = self.resnet18(image_tensor).squeeze(0)
                batch_embeddings.append(embedding)
            
            current_embeddings.append(torch.stack(batch_embeddings))
        current_embeddings = torch.stack(current_embeddings)

        batch_size = current_embeddings.size(0)
        # Stacking 4 current features
        # Stacking 4 goal features 
        current_cat = torch.cat([current_embeddings[:, i] for i in range(self.num_cameras)], dim=3)
        goal_cat    = torch.cat([goal_embeddings[:, i] for i in range(self.num_cameras)], dim=3)
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

    model = Resnet18Mlp5().to(device="cuda", dtype=torch.float)
    weight_path = ''
    model.load_weight(weight_path)
    model.set_goal(goal_images, "green chair.")

    output, _ = model([current_images])
    print(torch.argmax(output, dim=2))