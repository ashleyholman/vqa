import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from datetime import datetime

from tqdm import tqdm
from src.data.vqa_dataset import VQADataset
from src.models.vqa_model import VQAModel
from src.util.linear_warmup_scheduler import LinearWarmupScheduler

class ModelTrainer:
    def __init__(self, config, model: VQAModel, dataset, optimizer, num_dataloader_workers):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.isModelParallel = False
        self.unfreeze_every_epochs = self.config.finetune_unfreeze_every_epochs if self.config.finetune_unfreeze_every_epochs else 4

        if self.config.learning_rate_warmup_steps:
            # Learning rate scheduler requires 'initial_lr' set on all parameter groups in the model
            for param_group in optimizer.param_groups:
                param_group['initial_lr'] = self.config.learning_rate

            # create a learning rate scheduler to gradually warm up the learning
            # rate over the given number of steps
            if not isinstance(self.config.learning_rate_warmup_steps, int):
                raise ValueError("learning_rate_warmup_steps must be an integer.")

            self.lr_scheduler = LinearWarmupScheduler(optimizer, warmup_steps=self.config.learning_rate_warmup_steps)

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
        return DataLoader(self.dataset, batch_size=batch_size, num_workers=num_dataloader_workers, shuffle=self.config.shuffle)

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

    def train_one_epoch(self, epoch, performance_tracker=None, no_progress_bar=True):
        # Wrap data_loader with tqdm to show a progress bar, unless --no-progress-bar was specified
        dataloader = self.dataloader
        if not no_progress_bar:
            total_batches = len(dataloader)
            if self.config.max_batches_per_epoch:
                total_batches = min(self.config.max_batches_per_epoch, total_batches)
            dataloader = tqdm(dataloader, total=total_batches)

        if self.config.finetune_from_snapshot and self.config.finetune_gradual_unfreezing:
            # unfreeze layers according to the current epoch
            self.model.unfreeze_layers(1+int((epoch-1) / self.unfreeze_every_epochs))

        self.model.train()
        for idx, batch in enumerate(dataloader, start=1):
            # Transfer data to the appropriate device
            if self.config.finetune_from_snapshot:
                # send image and text features to the device
                batch["image"] = batch["image"].to(self.device)
                batch["input_ids"] = batch["input_ids"].to(self.device)
                batch["attention_mask"] = batch["attention_mask"].to(self.device)
            else:
                # send pre-computed embeddings to device
                batch["image_embedding"] = batch["image_embedding"].to(self.device)
                batch["question_embedding"] = batch["question_embedding"].to(self.device)

            labels = batch["label"].to(self.device)

            # Clear the gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(batch)
            loss = self.loss_function(logits, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Step the learning-rate scheduler, if we have one
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Use PerformanceTracker to track the model's accuracy, loss etc
            if performance_tracker:
                performance_tracker.update_metrics(logits, labels, loss.item())

            if idx % 500 == 0:
                print(f"\nBatch {idx}..")

            # Limit the number of batches per epoch if configured to do so
            if self.config.max_batches_per_epoch and idx >= self.config.max_batches_per_epoch:
                print(f"Reached max batches per epoch ({self.config.max_batches_per_epoch}).")
                break