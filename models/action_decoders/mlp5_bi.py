import torch
from torch import nn
from Object_Centric_Local_Navigation.models.modules.flash_cross_attention import FlashCrossAttention

class Mlp5Bi(nn.Module):

    def __init__(self, num_trunk_channels, pool_num):
        super().__init__()

        self.cross_attention = FlashCrossAttention(embed_dim=num_trunk_channels, num_heads=1)
        self.global_pool = nn.AdaptiveAvgPool2d((pool_num, pool_num*4))

        self.fc_layer1 = nn.Sequential(
            nn.Linear(num_trunk_channels*pool_num*pool_num*4*2, 1024),
            nn.SiLU()
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.SiLU()
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.SiLU()
        )
        self.fc_layer4 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.SiLU()
        )
        self.fc_layer_x = nn.Linear(1024, 3)
        self.fc_layer_y = nn.Linear(1024, 3)
        self.fc_layer_r = nn.Linear(1024, 3)

    def forward(self, current_embeddings, goal_embeddings):
        """
        current_embeddings shape B, C, H, W
        """
        batch_size = current_embeddings.shape[0]
        # Curr - Goal Cross- Attention 
        curr_goal_attenion, cg_attention_score = self.cross_attention(current_embeddings, goal_embeddings)
        curr_attended = current_embeddings + curr_goal_attenion

        # Goal - Current Cross- Attention
        goal_curr_attenion, gc_attention_score = self.cross_attention(goal_embeddings, current_embeddings)
        goal_attended = goal_embeddings + goal_curr_attenion

        # Average pooling
        curr_attended = self.global_pool(curr_attended)
        goal_attended = self.global_pool(goal_attended)

        # Concatenate current and goal features
        features = torch.cat([curr_attended, goal_attended], dim=1).reshape(batch_size, -1)

        # Fully connected layers.
        hidden = self.fc_layer1(features)
        hidden = self.fc_layer2(hidden)
        hidden = self.fc_layer3(hidden)
        hidden = self.fc_layer4(hidden)
        
        output_x = self.fc_layer_x(hidden)
        output_y = self.fc_layer_y(hidden)
        output_r = self.fc_layer_r(hidden)

        outputs = torch.stack([output_x, output_y, output_r], dim=1)
        return outputs, (cg_attention_score, gc_attention_score)
    