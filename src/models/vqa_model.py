import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTModel, BertModel
from transformers import BertTokenizer

class VQAModel(nn.Module):
    MODEL_NAME = "lr1e4_weighted_dropout_batchnorm_answerembeddings"

    def __init__(self, answer_classes_text, hidden_size=768):
        super().__init__()

        # To produce image embeddings, we'll use a pre-trained ViT (visual transformer)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # To produce text embeddings for the questions, we'll use a pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Our model will apply a transform to the two embeddings, concatenate them,
        # then apply a final layer to predict an answer class.
        self.vit_transform = nn.Linear(self.vit.config.hidden_size, hidden_size)
        self.bert_transform = nn.Linear(self.bert.config.hidden_size, hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

        # Batch Normalization Layer
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # Layer transforming embeddings for comparison with answer embeddings
        self.embedding_transform = nn.Linear(hidden_size * 2, hidden_size)

        # Compute and store answer embeddings
        print("Computing answer embeddings...")
        self.recompute_answer_embeddings(answer_classes_text)

    def recompute_answer_embeddings(self, answer_classes_text):
        inputs = self.tokenizer(answer_classes_text, return_tensors="pt", padding=True, truncation=True, max_length=50)
        with torch.no_grad():
            self.answer_embeddings = self.bert(**inputs).pooler_output

    def forward(self, images, input_ids, attention_mask):
        image_embeddings = self.vit_transform(self.vit(images).pooler_output)
        question_embeddings = self.bert_transform(self.bert(input_ids, attention_mask).pooler_output) 

        embeddings = torch.cat([image_embeddings, question_embeddings], dim=1)

        # Apply batch normalization and dropout to the embeddings before passing them to the embedding_transform layer
        embeddings = self.batch_norm(embeddings)
        embeddings = self.dropout(embeddings)

        embeddings = self.embedding_transform(embeddings)

        # Compute similarities between embeddings and answer classes
        output = torch.matmul(F.normalize(embeddings), F.normalize(self.answer_embeddings.T))

        return output