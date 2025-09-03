import torch
from torch import nn

class Decoder4(nn.Module):
    pool_num = 8

    def __init__(self, num_trunk_channels):
        super().__init__()

        # Box
        self.box_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU()
        )
        self.box_base_vector = nn.Parameter(torch.randn(4, 256))
        self.box_cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.box_fc_layer = nn.Sequential(
            nn.Linear(1024, num_trunk_channels * self.pool_num * self.pool_num),
            nn.SiLU()
        )

        # Embedding
        self.embed_cross_attention = nn.MultiheadAttention(embed_dim=num_trunk_channels, num_heads=1, batch_first=True)
        self.embed_base_vector = nn.Parameter(torch.randn(16 * 16, num_trunk_channels))
        self.embed_max_pool = nn.AdaptiveMaxPool2d((self.pool_num, self.pool_num))

        self.fc_layer1 = nn.Sequential(
            nn.Linear(num_trunk_channels * self.pool_num * self.pool_num * 2, 1024),
            nn.SiLU()
        )
        self.fc_layer2 = nn.Sequential(
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
        B, C, H, W = current_embeddings.shape

        # Get box features
        ## Encode boxes
        current_boxes = current_boxes.unsqueeze(-1)
        current_box_embeddings = self.box_encoder(current_boxes)
        goal_boxes = goal_boxes.unsqueeze(-1)
        goal_box_embeddings = self.box_encoder(goal_boxes)
        
        box_base_vector = self.box_base_vector.unsqueeze(0).expand(B, -1, -1)
        box_score_matrix, _ = self.box_cross_attention(current_box_embeddings, goal_box_embeddings, box_base_vector)
        box_features = box_score_matrix.reshape(B, -1)
        box_features = self.box_fc_layer(box_features)

        # Get embeddings featurs
        ## Flattening
        current_embeddings = current_embeddings.flatten(start_dim=2).permute(0, 2, 1)
        goal_embeddings = goal_embeddings.flatten(start_dim=2).permute(0, 2, 1)

        ## Base vector
        embed_base_vector = self.embed_base_vector.unsqueeze(0).expand(B, -1, -1)

        ## Attention
        embed_score_matrix, _ = self.embed_cross_attention(current_embeddings, goal_embeddings, embed_base_vector)
        embed_score_matrix = embed_score_matrix.permute(0, 2, 1).reshape(B, C, H, W)

        ## Max pooling
        embed_features = self.embed_max_pool(embed_score_matrix).reshape(B, -1)

        features = torch.cat([box_features, embed_features], dim=1)

        hidden = self.fc_layer1(features)
        hidden = self.fc_layer2(hidden)
        
        output_x = self.fc_layer_x(hidden)
        output_y = self.fc_layer_y(hidden)
        output_r = self.fc_layer_r(hidden)

        action = torch.stack([output_x, output_y, output_r], dim=1)

        return action, None