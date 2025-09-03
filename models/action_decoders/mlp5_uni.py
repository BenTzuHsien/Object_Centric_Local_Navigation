import torch
from torch import nn
from Object_Centric_Local_Navigation.models.modules.flash_cross_attention import FlashCrossAttention

class Mlp5Uni(nn.Module):

    def __init__(self, num_trunk_channels, pool_num):
        super().__init__()

        # Box
        self.box_fc_layer = nn.Sequential(
            nn.Linear(8, 512),
            nn.SiLU(),
            nn.Linear(512, num_trunk_channels*pool_num*pool_num*4),
            nn.SiLU(),
        )

        # Embedding
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

    def forward(self, current_boxes, current_embeddings, goal_boxes, goal_embeddings):
        """
        current_boxes shape(B, 4)
        current_embeddings shape(B, C', H', W')
        goal_boxes shape(B, 4)
        goal_embeddings shape(B, C', H', W')
        """
        batch_size = current_embeddings.shape[0]

        # Get box features
        box_features = torch.cat([current_boxes, goal_boxes], dim=1)
        box_features = self.box_fc_layer(box_features)

        # Get embeddings featurs
        ## Curr - Goal Cross- Attention 
        curr_goal_attenion, attention_score = self.cross_attention(current_embeddings, goal_embeddings)
        curr_attended = current_embeddings + curr_goal_attenion

        # Concatenate current and goal features
        embed_features = self.global_pool(curr_attended).reshape(batch_size, -1)

        features = torch.cat([box_features, embed_features], dim=1)

        # Fully connected layers.
        hidden = self.fc_layer1(features)
        hidden = self.fc_layer2(hidden)
        hidden = self.fc_layer3(hidden)
        hidden = self.fc_layer4(hidden)
        
        output_x = self.fc_layer_x(hidden)
        output_y = self.fc_layer_y(hidden)
        output_r = self.fc_layer_r(hidden)

        outputs = torch.stack([output_x, output_y, output_r], dim=1)
        return outputs, attention_score
    