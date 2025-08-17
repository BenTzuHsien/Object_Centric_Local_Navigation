import torch
from torch import nn

class Decoder1(nn.Module):
    pool_num = 8

    def __init__(self, patch_num, num_trunk_channels):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(embed_dim=num_trunk_channels, num_heads=1, batch_first=True)
        self.base_vector = nn.Parameter(torch.randn(num_trunk_channels, patch_num, patch_num * 4))
        self.max_pool = nn.AdaptiveMaxPool2d((self.pool_num, self.pool_num * 4))

        self.fc_layer1 = nn.Sequential(
            nn.Linear(num_trunk_channels * self.pool_num * self.pool_num * 4, 1024),
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
        B, C, H, W = current_embeddings.shape

        # Normalize
        current_embeddings = nn.functional.normalize(current_embeddings, dim=1)
        goal_embeddings = nn.functional.normalize(goal_embeddings, dim=1)

        # Flattening
        current_embeddings = current_embeddings.flatten(start_dim=2).permute(0, 2, 1)
        goal_embeddings = goal_embeddings.flatten(start_dim=2).permute(0, 2, 1)

        # Base Vector
        base_vector = self.base_vector.unsqueeze(0).expand(B, -1, -1, -1)
        base_vector = base_vector.flatten(start_dim=2).permute(0, 2, 1)

        # Attention
        score_matrix, attention_score = self.cross_attention(current_embeddings, goal_embeddings, base_vector)
        score_matrix = score_matrix.permute(0, 2, 1).reshape(B, C, H, W)

        # Max Pooling
        features = self.max_pool(score_matrix).reshape(B, -1)

        hidden = self.fc_layer1(features)
        hidden = self.fc_layer2(hidden)
        hidden = self.fc_layer3(hidden)
        hidden = self.fc_layer4(hidden)
        
        output_x = self.fc_layer_x(hidden)
        output_y = self.fc_layer_y(hidden)
        output_r = self.fc_layer_r(hidden)

        action = torch.stack([output_x, output_y, output_r], dim=1)

        return action, (score_matrix, attention_score)