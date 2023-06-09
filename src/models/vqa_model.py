import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, ViTModel

from src.data.embeddings_manager import EmbeddingsManager
from src.models.gated_multi_model_unit import GatedMultiModalUnit
from src.models.model_configuration import ModelConfiguration

class VQAModel(nn.Module):
    def __init__(self, config: ModelConfiguration, embeddings_manager: EmbeddingsManager, answer_classes_text=None):
        super().__init__()
        self.config = config

        if config.finetune_from_snapshot:
            # When finetuning, we include the full bert and vit models in our model
            self.bert = BertModel.from_pretrained(self.config.input_embedding_model_names['text'])
            self.vit = ViTModel.from_pretrained(self.config.input_embedding_model_names['vision'])

            if self.config.finetune_gradual_unfreezing:
                # If finetuning with gradual unfreezing, we begin with all layers frozen in vit/bert.
                for param in self.bert.parameters():
                    param.requires_grad = False
                for param in self.vit.parameters():
                    param.requires_grad = False

        self.image_embedding_size = embeddings_manager.get_embedding_size('vision')
        self.text_embedding_size = embeddings_manager.get_embedding_size('text')

        if self.config.transform_input_embeddings:
          transformed_image_embedding_size, transformed_text_embedding_size = self.get_embeddings_transform_output_sizes()
          self.image_transform = nn.Linear(self.image_embedding_size, transformed_image_embedding_size)
          self.question_transform = nn.Linear(self.text_embedding_size, transformed_text_embedding_size)

          print(f"Transforming image embeddings from {self.image_embedding_size} to {transformed_image_embedding_size}")
          print(f"Transforming text embeddings from {self.text_embedding_size} to {transformed_text_embedding_size}")

          # update self.image_embedding_size / self.text_embedding_size to reflect the transformed sizes
          self.image_embedding_size = transformed_image_embedding_size
          self.text_embedding_size = transformed_text_embedding_size

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

    def get_embeddings_transform_output_sizes(self):
        if self.config.transform_input_embeddings_to_size == "smallest":
            image_output_size = text_output_size = min(self.image_embedding_size, self.text_embedding_size)
        elif self.config.transform_input_embeddings_to_size == "largest":
            image_output_size = text_output_size = max(self.image_embedding_size, self.text_embedding_size)
        elif self.config.transform_input_embeddings_to_size == "preserve":
            image_output_size = self.image_embedding_size
            text_output_size = self.text_embedding_size
        elif isinstance(self.config.transform_input_embeddings_to_size, int):
            image_output_size = text_output_size = self.config.transform_input_embeddings_to_size
        else:
            raise ValueError("Invalid value for transform_input_embeddings_to_size: {}".format(self.config.transform_input_embeddings_to_size))

        return image_output_size, text_output_size

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

    def forward(self, input):
        if self.config.finetune_from_snapshot:
          image_embeddings = self.vit(input['image']).pooler_output
          question_embeddings = self.bert(input_ids=input['input_ids'], attention_mask=input['attention_mask']).pooler_output
        else:
            # when not finetuning, our input is pre-computed embeddings from
            # vit/bert which is an optimisation that saves having to compute the
            # vit/bert embeddings in the forward pass
            image_embeddings = input['image_embedding']
            question_embeddings = input['question_embedding']

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

    def unfreeze_layers(self, num_layers):
        for i, layer in enumerate(reversed(self.bert.encoder.layer)):
            if i < num_layers:
                print(f"Unfreezing BERT layer {i}")
                for param in layer.parameters():
                    param.requires_grad = True

        for i, layer in enumerate(reversed(self.vit.encoder.layer)):
            if i < num_layers:
                print(f"Unfreezing ViT layer {i}")
                for param in layer.parameters():
                    param.requires_grad = True