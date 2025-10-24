import torch
from torch import nn

class ScoreMlp5(nn.Module):
    pool_num = 8
    embed_dim = 128  # Output dimension of per-row MLP

    def __init__(self):
        super().__init__()

        # Box
        self.box_fc_layer = nn.Sequential(
            nn.Linear(8, 512),
            nn.SiLU(),
            nn.Linear(512, self.pool_num ** 4),
            nn.SiLU(),
        )

        # Embedding
        self.num_tokens = self.pool_num * self.pool_num * 4  # Total spatial tokens after pooling
        self.embed_avg_pool = nn.AdaptiveAvgPool2d((self.pool_num, self.pool_num))

        self.fc_layer1 = nn.Sequential(
            nn.Linear(self.pool_num ** 4 * 2, 1024),
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
        ## Normalize
        norm_current_embeddings = nn.functional.normalize(current_embeddings, dim=1)
        norm_goal_embeddings = nn.functional.normalize(goal_embeddings, dim=1)

        ## Average Pooling
        norm_current_embeddings = self.embed_avg_pool(norm_current_embeddings)
        norm_goal_embeddings = self.embed_avg_pool(norm_goal_embeddings)

        ## Flattening
        current_embeddings = norm_current_embeddings.flatten(start_dim=2).permute(0, 2, 1)
        goal_embeddings = norm_goal_embeddings.flatten(start_dim=2).permute(0, 2, 1)
        
        ## Cosine similarity
        score_matrix = current_embeddings @ goal_embeddings.transpose(-2, -1)
        embed_features = score_matrix.reshape(batch_size, -1)

        features = torch.cat([box_features, embed_features], dim=1)

        # Fully connected layers.
        hidden = self.fc_layer1(features)
        hidden = self.fc_layer2(hidden)
        hidden = self.fc_layer3(hidden)
        hidden = self.fc_layer4(hidden)
        
        output_x = self.fc_layer_x(hidden)
        output_y = self.fc_layer_y(hidden)
        output_r = self.fc_layer_r(hidden)

        action = torch.stack([output_x, output_y, output_r], dim=1)

        return action, score_matrix