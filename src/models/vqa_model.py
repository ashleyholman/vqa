import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel

from src.data.embeddings_manager import EmbeddingsManager
from src.models.gated_multi_model_unit import GatedMultiModalUnit
from src.models.model_configuration import ModelConfiguration

class VQAModel(nn.Module):
    def __init__(self, config: ModelConfiguration, embeddings_manager: EmbeddingsManager, answer_classes_text=None):
        super().__init__()
        self.config = config

        self.image_embedding_size = embeddings_manager.get_embedding_size('vision')
        self.text_embedding_size = embeddings_manager.get_embedding_size('text')

        if self.config.transform_input_embeddings:
          self.image_transform = nn.Linear(self.image_embedding_size, self.image_embedding_size)
          self.question_transform = nn.Linear(self.text_embedding_size, self.text_embedding_size)

        if self.config.use_gating:
            self.gated_unit = GatedMultiModalUnit(self.image_embedding_size, self.text_embedding_size)

        if self.config.use_dropout:
            self.dropout_input = nn.Dropout(self.config.dropout_input_probability)
            self.dropout_hidden = nn.Dropout(self.config.dropout_hidden_probability)

        if self.config.use_batch_normalization:
            self.batch_norm = nn.BatchNorm1d(self.image_embedding_size + self.text_embedding_size)

        if self.config.num_hidden_layers > 0:
            self.hidden_layers = self._build_hidden_layers()

        # Determine the input and output sizes of the final layer (head)
        # The input size will depend on the output size of the layer that
        # preceeds it.
        # The output size will depend on whether we're predicting an output
        # class index directly, or producing an embedding to compare against
        # answer embeddings
        if self.config.num_hidden_layers > 0:
            head_input_size = self.config.hidden_size
        else:
            head_input_size = self.image_embedding_size + self.text_embedding_size

        if self.config.use_answer_embeddings:
            # when using answer embeddings, our head layer outputs am embedding which
            # will be compared against the embeddings of the answer class labels (text).
            head_output_size = self.text_embedding_size
            self.recompute_answer_embeddings(answer_classes_text)
        else:
            head_output_size = len(answer_classes_text)

        print("head_input_size:", head_input_size)
        print("head_output_size:", head_output_size)
        self.head = nn.Linear(head_input_size, head_output_size)

    def _build_hidden_layers(self):
        # Add first hidden layer (input size is different, since it takes the embeddings as input)
        layers = [nn.Linear(self.image_embedding_size + self.text_embedding_size, self.config.hidden_size)]
        if self.config.use_batch_normalization:
            layers.append(nn.BatchNorm1d(self.config.hidden_size))
        layers.append(nn.ReLU())
        if self.config.use_dropout:
            layers.append(self.dropout_hidden)

        # Add any additional hidden layers.  Their input size is self.config.hidden_size.
        for _ in range(self.config.num_hidden_layers - 1):
            layers.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
            if self.config.use_batch_normalization:
                layers.append(nn.BatchNorm1d(self.config.hidden_size))
            layers.append(nn.ReLU())
            if self.config.use_dropout:
                layers.append(self.dropout_hidden)
        return nn.Sequential(*layers)

    def recompute_answer_embeddings(self, answer_classes_text):
        bert = BertModel.from_pretrained(self.config.input_embedding_model_names['text'])
        tokenizer = BertTokenizer.from_pretrained(self.config.input_embedding_model_names['text'])

        inputs = tokenizer(answer_classes_text, return_tensors="pt", padding=True, truncation=True, max_length=50)
        with torch.no_grad():
            answer_embeddings = bert(**inputs).pooler_output
            if self.config.use_answer_embedding_z_normalization:
                print("Using z-normalization for answer embeddings")
                answer_embeddings -= answer_embeddings.mean(dim=0)
                answer_embeddings /= answer_embeddings.std(dim=0)
            else:
                # use regular vector normalization
                answer_embeddings = F.normalize(answer_embeddings)
            self.answer_embeddings = answer_embeddings

    def forward(self, image_embeddings, question_embeddings):
        if self.config.transform_input_embeddings:
          image_embeddings = self.image_transform(image_embeddings)
          question_embeddings = self.question_transform(question_embeddings)

        if self.config.use_gating:
            # use a weighted combination of the image and question embeddings
            embeddings = self.gated_unit(image_embeddings, question_embeddings)
        else:
            # simple concatenation of image and question embeddings
            embeddings = torch.cat([image_embeddings, question_embeddings], dim=1)

        if self.config.use_batch_normalization:
            embeddings = self.batch_norm(embeddings)

        if self.config.use_dropout:
            embeddings = self.dropout_input(embeddings)

        if self.config.num_hidden_layers > 0:
            embeddings = self.hidden_layers(embeddings)

        output = self.head(embeddings)

        if self.config.use_answer_embeddings:
            output = torch.matmul(F.normalize(output), self.answer_embeddings.T)

        return output