import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from datetime import datetime

from tqdm import tqdm
from src.data.vqa_dataset import VQADataset
from src.models.vqa_model import VQAModel

class ModelTrainer:
    def __init__(self, config, model, dataset, optimizer, num_dataloader_workers):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.isModelParallel = False

        print(f"Torch device: {self.device}")
        print(f"Using {num_dataloader_workers} DataLoader workers")

        # send the model to the GPU if we have one
        self._device_prep()

        self.dataloader = self._create_dataloader(num_dataloader_workers)

        self.loss_function = self._create_loss_function()

        # set model to training mode
        self.model.train()

        print("Trainer initialised.")

    def _device_prep(self):
        # If there's a GPU available...
        if torch.cuda.is_available():
            print('Training on GPU...')
            # Tell PyTorch to use the GPU.
            self.model = self.model.to(self.device)
            if self.config.use_answer_embeddings:
                # FIXME: Make this handled in snapshot manager or VQAModel class
                self.model.answer_embeddings = self.model.answer_embeddings.to(self.device)
            # If multiple GPUs are available, wrap model with DataParallel
            if torch.cuda.device_count() > 1:
                print(f"Let's use {torch.cuda.device_count()} GPUs!")
                self.model = nn.DataParallel(self.model)
                self.isModelParallel = True
        else:
            print('Training on CPU...')

    def _create_dataloader(self, num_dataloader_workers):
        # Create a DataLoader to handle batching of the dataset
        print("Loading dataset..")
        batch_size = self.config.batch_size
        if self.isModelParallel:
            # multiply the batch size by number of GPUs
            batch_size = batch_size * torch.cuda.device_count()
        print(f"Using batch size: {batch_size}")
        return DataLoader(self.dataset, batch_size=batch_size, num_workers=num_dataloader_workers, shuffle=True)

    def _create_loss_function(self):
        loss_args = {}
        if self.config.use_class_weights:
            print("Using class weights.")
            ### Use class weighting to counteract class imbalance
            class_counts = torch.Tensor(self.dataset.class_counts)
            # if any classes have 0 count, set them to 1 to avoid dividing by 0
            class_counts[class_counts == 0] = 1
            # take the reciprocal of the class counts so that the most frequent class has the lowest weight
            class_weights = 1. / class_counts
            # normalize weights so they sum to 1
            class_weights = class_weights / class_weights.sum()
            class_weights = class_weights.to(self.device)
            loss_args['weight'] = class_weights

        return torch.nn.CrossEntropyLoss(**loss_args)

    def train_one_epoch(self, performance_tracker=None):
        self.model.train()
        for idx, batch in enumerate(self.dataloader, start=1):
            # Transfer data to the appropriate device
            question_embeddings = batch["question_embedding"].to(self.device)
            image_embeddings = batch["image_embedding"].to(self.device)
            labels = batch["label"].to(self.device)

            # Clear the gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(image_embeddings, question_embeddings)
            loss = self.loss_function(logits, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Use PerformanceTracker to track the model's accuracy, loss etc
            if performance_tracker:
                performance_tracker.update_metrics(logits, labels, loss.item())