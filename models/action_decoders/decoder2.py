import torch
from torch import nn

class Decoder2(nn.Module):
    pool_num = 8
    embed_dim = 128  # Output dimension of per-row MLP

    def __init__(self):
        super().__init__()

        self.num_tokens = self.pool_num * self.pool_num * 4  # Total spatial tokens after pooling
        self.max_pool = nn.AdaptiveMaxPool2d((self.pool_num, self.pool_num*4))

        # Learnable positional encodings (one per query patch row)
        self.row_pos_encoding = nn.Parameter(torch.randn(1, self.num_tokens, self.num_tokens))

        # Shared MLP applied to each row of the score matrix
        self.row_mlp = nn.Sequential(
            nn.Linear(self.num_tokens, 256),
            nn.SiLU(),
            nn.Linear(256, self.embed_dim),
            nn.SiLU()
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.SiLU()
        )
        self.fc_layer_x = nn.Linear(128, 3)
        self.fc_layer_y = nn.Linear(128, 3)
        self.fc_layer_r = nn.Linear(128, 3)
        
    def forward(self, current_embeddings, goal_embeddings):
        """
        current_embeddings shape B, C, H, W
        """
        batch_size = current_embeddings.shape[0]

        # Normalize
        norm_current_embeddings = nn.functional.normalize(current_embeddings, dim=1)
        norm_goal_embeddings = nn.functional.normalize(goal_embeddings, dim=1)

        # Max Pooling
        norm_current_embeddings = self.max_pool(norm_current_embeddings)
        norm_goal_embeddings = self.max_pool(norm_goal_embeddings)

        # Flattening
        norm_current_embeddings = norm_current_embeddings.flatten(start_dim=2).permute(0, 2, 1)
        norm_goal_embeddings = norm_goal_embeddings.flatten(start_dim=2).permute(0, 2, 1)
        
        score_matrix = norm_current_embeddings @ norm_goal_embeddings.transpose(-2, -1)
        score_matrix_encoded = score_matrix + self.row_pos_encoding   # Add learnable positional encoding (broadcasts over batch)

         # Apply shared row-wise MLP
        row_input = score_matrix_encoded.view(batch_size * self.num_tokens, self.num_tokens)              # (B * HW, HW)
        row_output = self.row_mlp(row_input)                   # (B * HW, embed_dim)
        row_output = row_output.view(batch_size, self.num_tokens, self.embed_dim)    # (B, HW, embed_dim)
        features = row_output.mean(dim=1)

        hidden = self.fc_layer(features)
        
        output_x = self.fc_layer_x(hidden)
        output_y = self.fc_layer_y(hidden)
        output_r = self.fc_layer_r(hidden)

        action = torch.stack([output_x, output_y, output_r], dim=1)

        return action, score_matrix