import torch
import torch.nn as nn

from transformers import ViTModel, BertModel

class VQAModel(nn.Module):
    def __init__(self, num_answer_classes, hidden_size=768):
        super().__init__()

        vit_model_name = "google/vit-base-patch16-224-in21k"  # name of the ViT model
        self.vit = ViTModel.from_pretrained(vit_model_name)

        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Create the layers
        self.vit_transform = nn.Linear(self.vit.config.hidden_size, hidden_size)
        self.bert_transform = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size * 2, num_answer_classes)

    def forward(self, images, input_ids, attention_mask):
        image_embeddings = self.vit_transform(self.vit(images).pooler_output)
        question_embeddings = self.bert_transform(self.bert(input_ids, attention_mask).pooler_output) 
        embeddings = torch.cat([image_embeddings, question_embeddings], dim=1)
        logits = self.head(embeddings)
        return logits
