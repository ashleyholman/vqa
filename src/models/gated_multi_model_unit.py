import torch.nn as nn
import torch

class GatedMultiModalUnit(nn.Module):
    def __init__(self, image_embedding_size, text_embedding_size):
        super().__init__()
        self.image_gate = nn.Sequential(
            nn.Linear(image_embedding_size + text_embedding_size, image_embedding_size),
            nn.Sigmoid()
        )
        self.text_gate = nn.Sequential(
            nn.Linear(image_embedding_size + text_embedding_size, text_embedding_size),
            nn.Sigmoid()
        )

    def forward(self, image_embeddings, text_embeddings):
        combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=1)
        image_gate_values = self.image_gate(combined_embeddings)
        text_gate_values = self.text_gate(combined_embeddings)

        # Apply gating to each embedding
        image_weighted = image_embeddings * image_gate_values
        text_weighted = text_embeddings * text_gate_values

        return torch.cat([image_weighted, text_weighted], dim=1)