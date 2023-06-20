import torch.nn as nn
import torch

class GatedMultiModalUnit(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.image_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        self.text_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, image_embeddings, text_embeddings):
        image_gate_values = self.image_gate(image_embeddings)
        text_gate_values = self.text_gate(text_embeddings)

        gate_sum = image_gate_values + text_gate_values

        image_weighted = image_embeddings * (image_gate_values / gate_sum)
        text_weighted = text_embeddings * (text_gate_values / gate_sum)

        return torch.cat([image_weighted, text_weighted], dim=-1)