import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, ViTModel, ViTImageProcessor
from src.models.model_configuration import ModelConfiguration

class BertTokenizingDataset(Dataset):
    def __init__(self, config, texts):
        self.texts = texts
        self.tokenizer = BertTokenizer.from_pretrained(config.input_embedding_model_names['text'])

    def __getitem__(self, index):
        text_input = self.texts[index]
        encoding = self.tokenizer.encode_plus(
                    text_input,
                    max_length=30,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )

        return { 'input_ids': encoding['input_ids'].flatten(),
                 'attention_mask': encoding['attention_mask'].flatten() }

    def __len__(self):
        return len(self.texts)


class ViTPreProcessingDataset(Dataset):
    def __init__(self, config, image_paths):
        self.image_paths = image_paths
        self.vit_preprocessor = ViTImageProcessor.from_pretrained(config.input_embedding_model_names['vision'])


    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = np.array(image)
        features = self.vit_preprocessor(image, return_tensors='pt')
        return { 'pixel_values' : features['pixel_values'].squeeze(0) }

    def __len__(self):
        return len(self.image_paths)


class EmbeddingsManager:
    '''
    Generates embeddings for a given dataset, caches them on disk and provides
    them to callers.
    '''
    def __init__(self, config: ModelConfiguration, dataset_type, num_dataloader_workers):
        self.config = config
        self.dataset_type = dataset_type
        self.num_dataloader_workers = num_dataloader_workers
        self.DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"EmbeddingsManager using {self.num_dataloader_workers} DataLoader workers.")

    def __get_embeddings_file_path(self, modality) -> str:
        model_name = self.config.input_embedding_model_names[modality].split('/')[-1]
        return os.path.join(self.DATA_DIR, f'{self.dataset_type}_{modality}_embeddings.{model_name}.pt')

    def get_embeddings(self, modality, inputs) -> torch.Tensor:
        file_path = self.__get_embeddings_file_path(modality)
        if os.path.exists(file_path):
            print(f'Loading {modality} embeddings from {file_path}')
            loaded_embeddings = torch.load(file_path)
            # loaded embeddings size should be of length of inputs
            if len(loaded_embeddings) != len(inputs):
                raise ValueError(f'Loaded embeddings size ({len(loaded_embeddings)}) does not match inputs size ({len(inputs)})')
            return loaded_embeddings
        else:
            # Stored embeddings aren't present on disk.  Generate them now.
            print(f'No {modality} embeddings found on disk.  Generating...')
            return self.generate_and_save_text_embeddings(modality, inputs)

    def generate_and_save_text_embeddings(self, modality, inputs) -> None:
        if modality == 'text':
            dataset = BertTokenizingDataset(self.config, inputs)
            model = BertModel.from_pretrained(self.config.input_embedding_model_names['text'])
        elif modality == 'vision':
            dataset = ViTPreProcessingDataset(self.config, inputs)
            model = ViTModel.from_pretrained(self.config.input_embedding_model_names['vision'])
        else:
            raise ValueError(f'Unsupported modality: {modality}')

        dataloader = DataLoader(dataset, batch_size=32, num_workers=self.num_dataloader_workers)

        save_path = self.__get_embeddings_file_path(modality)
        print(f'Generating and saving {modality} embeddings to {save_path}')

        model.eval()
        with torch.no_grad():
            embeddings = []
            for batch in dataloader:
                output = model(**batch)
                embeddings.append(output['pooler_output'])
            embeddings = torch.cat(embeddings)
            torch.save(embeddings, save_path)
            return embeddings