import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel

class VQAModel(nn.Module):
    MODEL_NAME = "lr1e2_weighted_dropout_batchnorm_answerembeddings_inputembeddings_hidden2"

    def __init__(self, answer_classes_text, hidden_size=768):
        super().__init__()

        # Our model will apply a transform to each embedding before concatenating them
        self.image_transform = nn.Linear(hidden_size, hidden_size)
        self.question_transform = nn.Linear(hidden_size, hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

        # Batch Normalization Layer
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # Layer transforming embeddings for comparison with answer embeddings
        self.embedding_transform = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Compute and store answer embeddings
        print("Computing answer embeddings...")
        self.recompute_answer_embeddings(answer_classes_text)

    def recompute_answer_embeddings(self, answer_classes_text):
        bert = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        inputs = tokenizer(answer_classes_text, return_tensors="pt", padding=True, truncation=True, max_length=50)
        with torch.no_grad():
            self.answer_embeddings = bert(**inputs).pooler_output

    def forward(self, image_embeddings, question_embeddings):
        image_embeddings = self.image_transform(image_embeddings)
        question_embeddings = self.question_transform(question_embeddings)

        embeddings = torch.cat([image_embeddings, question_embeddings], dim=1)

        # Apply batch normalization and dropout to the embeddings before passing them to the embedding_transform layer
        embeddings = self.batch_norm(embeddings)
        embeddings = self.dropout(embeddings)

        embeddings = self.embedding_transform(embeddings)

        # Compute similarities between embeddings and answer classes
        output = torch.matmul(F.normalize(embeddings), F.normalize(self.answer_embeddings.T))

        return output