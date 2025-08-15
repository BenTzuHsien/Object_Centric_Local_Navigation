import torch
from torch import nn

class Decoder1(nn.Module):
    num_top_matches = 1024

    def __init__(self):
        super().__init__()

        self.fc_layer1 = nn.Sequential(
            nn.Linear(self.num_top_matches * 3, 1024),
            nn.SiLU()
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.SiLU()
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU()
        )
        self.fc_layer4 = nn.Sequential(
            nn.Linear(64, 16),
            nn.SiLU()
        )
        self.fc_layer_x = nn.Linear(16, 3)
        self.fc_layer_y = nn.Linear(16, 3)
        self.fc_layer_r = nn.Linear(16, 3)
        
    def forward(self, current_embeddings, goal_embeddings):
        """
        current_embeddings shape B, C, H, W
        """
        norm_current_embeddings = nn.functional.normalize(current_embeddings, dim=1).flatten(start_dim=2).permute(0, 2, 1)
        norm_goal_embeddings = nn.functional.normalize(goal_embeddings, dim=1).flatten(start_dim=2).permute(0, 2, 1)
        
        score_matrix = norm_current_embeddings @ norm_goal_embeddings.transpose(-2, -1)
        B, Hs, Ws = score_matrix.shape
        values, indices = torch.topk(score_matrix.flatten(start_dim=1), k=self.num_top_matches)
        order = indices.argsort(dim=1)
        values = torch.gather(values,  dim=1, index=order)
        indices = torch.gather(indices,  dim=1, index=order)
        rows = indices // Ws
        cols = indices % Ws
        top_matches_pairs = torch.stack([rows, cols, values], dim=-1)

        hidden = self.fc_layer1(top_matches_pairs.flatten(start_dim=1))
        hidden = self.fc_layer2(hidden)
        hidden = self.fc_layer3(hidden)
        hidden = self.fc_layer4(hidden)
        
        output_x = self.fc_layer_x(hidden)
        output_y = self.fc_layer_y(hidden)
        output_r = self.fc_layer_r(hidden)

        action = torch.stack([output_x, output_y, output_r], dim=1)

        return action, score_matrix