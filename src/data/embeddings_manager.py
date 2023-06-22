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
    def __init__(self, config: ModelConfiguration, num_dataloader_workers):
        self.config = config
        self.num_dataloader_workers = num_dataloader_workers
        self.DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_sizes = {}

        print(f"EmbeddingsManager using {self.num_dataloader_workers} DataLoader workers.")

    def __get_embeddings_file_path(self, dataset_type, modality) -> str:
        model_name = self.config.input_embedding_model_names[modality].split('/')[-1]
        return os.path.join(self.DATA_DIR, f'{dataset_type}_{modality}_embeddings.{model_name}.pt')

    def get_embeddings(self, dataset_type, modality, inputs) -> torch.Tensor:
        file_path = self.__get_embeddings_file_path(dataset_type, modality)
        if os.path.exists(file_path):
            print(f'Loading {modality} embeddings from {file_path}')
            loaded_data = torch.load(file_path, map_location=torch.device('cpu'))
            embeddings = loaded_data['embeddings']
            input_indices = loaded_data['input_indices']
            # reconstruct the embeddings with duplicates in the order of the original input list
            input_embeddings = embeddings[input_indices]
            # loaded embeddings size should be of length of inputs
            if len(input_embeddings) != len(inputs):
                raise ValueError(f'Loaded embeddings size ({len(input_embeddings)}) does not match inputs size ({len(inputs)})')
            return input_embeddings
        else:
            # Stored embeddings aren't present on disk.  Generate them now.
            model_name = self.config.input_embedding_model_names[modality].split('/')[-1]
            print(f'{modality} embeddings for {dataset_type} dataset not found on disk for model "{model_name}".  Generating...')
            return self.generate_and_save_embeddings(dataset_type, modality, inputs)

    def generate_and_save_embeddings(self, dataset_type, modality, inputs) -> None:
        # de-duplicate the inputs so that we avoid processing the same inputs multiple times.
        # this will save a lot of time, eg. VQA training dataset size is 443k
        # questions, but there are only 82k unique images.
        # In order to generate an embeddings list that matches the original
        # dataset, we store the input indices as well
        unique_inputs, input_indices = np.unique(inputs, return_inverse=True)
        print(f"Original dataset inputs deduped from {len(inputs)} down to {len(unique_inputs)} inputs ({(1-(len(unique_inputs)/len(inputs))) * 100:.2f}% reduction).")

        if modality == 'text':
            dataset = BertTokenizingDataset(self.config, unique_inputs)
            model = BertModel.from_pretrained(self.config.input_embedding_model_names['text']).to(self.device)
        elif modality == 'vision':
            dataset = ViTPreProcessingDataset(self.config, unique_inputs)
            model = ViTModel.from_pretrained(self.config.input_embedding_model_names['vision']).to(self.device)
        else:
            raise ValueError(f'Unsupported modality: {modality}')

        dataloader = DataLoader(dataset, batch_size=32, num_workers=self.num_dataloader_workers)

        save_path = self.__get_embeddings_file_path(dataset_type, modality)
        print(f'Generating and saving {modality} embeddings for {dataset_type} dataset to {save_path}...')

        model.eval()
        with torch.no_grad():
            embeddings = []
            total_batches = len(dataloader)
            batch_counter = 0
            for batch in dataloader:
                # send batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # run model
                output = model(**batch)
                embeddings.append(output['pooler_output'])

                batch_counter += 1
                if batch_counter % 50 == 0:
                    print(f"{batch_counter}/{total_batches} batches processed...")

            embeddings = torch.cat(embeddings).cpu()

            # save both the unique embeddings and the indices needed for
            # reconstruting an embeddings list that matches the original
            # the dataset's inputs.
            torch.save({'embeddings': embeddings, 'input_indices': input_indices}, save_path)

            # reconstruct the embeddings with duplicates
            input_embeddings = embeddings[input_indices]

            return input_embeddings

    def get_embedding_size(self, modality):
        if modality in self.embedding_sizes:
            # If the size is already in the cache, return it
            return self.embedding_sizes[modality]

        # If the size isn't in the cache, we need to calculate it
        if modality == 'text':
            model = BertModel.from_pretrained(self.config.input_embedding_model_names['text'])
            dummy_input = torch.zeros(1, model.config.max_position_embeddings).long().to(self.device)  # Creating a dummy input
        elif modality == 'vision':
            model = ViTModel.from_pretrained(self.config.input_embedding_model_names['vision'])
            dummy_input = torch.zeros(1, 3, model.config.image_size, model.config.image_size).to(self.device)  # Creating a dummy input
        else:
            raise ValueError(f'Unsupported modality: {modality}')

        model.eval().to(self.device)
        with torch.no_grad():
            output = model(dummy_input)
            embedding_size = output['pooler_output'].size(-1)

        # Cache the size for future use
        self.embedding_sizes[modality] = embedding_size
        print(f"EmbeddingsManager: {modality} embedding size: {embedding_size}")

        return embedding_size