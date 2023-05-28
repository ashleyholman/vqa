import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from datetime import datetime

from tqdm import tqdm
from src.data.vqa_dataset import VQADataset
from src.models.vqa_model import VQAModel

from src.snapshots.vqa_snapshot_manager import VQASnapshotManager
from src.snapshots.snapshot import Snapshot
from src.metrics.metrics_manager import MetricsManager
from src.metrics.performance_tracker import PerformanceTracker

class ModelTrainer:
    METRICS_SOURCE = "train_model"

    def __init__(self, config, num_dataloader_workers, dataset_type, snapshot_name=None):
        self.config = config
        self.dataset_type = dataset_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.isModelParallel = False

        self.snapshot_manager = VQASnapshotManager()
        self.metrics_manager = MetricsManager(self.METRICS_SOURCE)

        print(f"Torch device: {self.device}")
        print(f"Using {num_dataloader_workers} DataLoader workers")

        if snapshot_name is not None:
            self.model, self.dataset, self.optimizer, self.trained_until_epoch = self._load_snapshot(snapshot_name)
            print(f"Will resume training from epoch {self.start_epoch}.")
        else:
            self.model, self.dataset, self.optimizer = self._initialise_untrained_model()
            self.trained_until_epoch = 0

        # send the model to the GPU if we have one
        self._device_prep()

        self.dataloader = self._create_dataloader(num_dataloader_workers)

        self.loss_function = self._create_loss_function()

        # set model to training mode
        self.model.train()

        print("Trainer initialised.")

    def _load_snapshot(self, snapshot_name):
        # snapshot_manager will throw an exception of the snapshot doesn't exist.  caller should handle that.
        snapshot = self.snapshot_manager.load_snapshot(snapshot_name, self.dataset_type, self.device)

        # Since we're resuming training, the dataset and model version we're training
        # on should be the same as the ones in the snapshot, otherwise we'll get
        # unexpected results.
        if (snapshot.get_metadata()["settype"] != self.dataset_type):
            print(f"ERROR: Dataset type '{self.dataset_type}' does not match the dataset type in the snapshot '{snapshot.get_metadata()['dataset_type']}'.")
            return

        if (snapshot.get_metadata()["model_version"] != self.config.model_name):
            print(f"ERROR: Model name '{self.config.model_name}' does not match the model name in the snapshot '{snapshot.get_metadata()['model_name']}'.")
            return

        print(f"Using snapshot: {snapshot_name}")
        dataset = snapshot.get_dataset()
        model = snapshot.get_model()
        optimizer = snapshot.get_optimizer()

        if (snapshot.isLightweight()):
            print(f"WARNING: Snapshot '{snapshot_name}' is lightweight, meaning the pretrained "
                  "weights of the BERT and ViT models are not included in the snapshot.  Those models "
                  "will be initialized from huggingface.  If the underlying weights of these models "
                  "have changed since the snapshot was created, the model may not train correctly.")

        return model, dataset, optimizer, snapshot.get_metadata()["epoch"]

    def _initialise_untrained_model(self):
        # load dataset
        dataset = VQADataset(self.dataset_type)

        # load model
        print("Answer classes: ", dataset.answer_classes)
        model = VQAModel(dataset.answer_classes)

        # Create a new optimizer
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)

        return model, dataset, optimizer

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
            class_weights = class_weights.to(device)
            loss_args['weight'] = class_weights

        return torch.nn.CrossEntropyLoss(**loss_args)

    def train(self, until_epoch, skip_s3_storage=False, lightweight_snapshots=False, no_progress_bar=False):
        if (self.trained_until_epoch >= until_epoch):
            # We've already trained up until the requested epoch.  Nothing to do.
            print(f"Snapshot '{snapshot_name}' already trained for {self.trained_until_epoch} epochs.  Nothing to do.")
            return

        if skip_s3_storage:
            print("WARNING: Skipping S3 storage of snapshots.  Snapshots will only be stored locally.")

        start_epoch = self.trained_until_epoch + 1

        print("Beginning training.")
        performance_tracker = PerformanceTracker()

        for epoch in range(start_epoch, until_epoch+1):
            is_snapshot_epoch = (epoch % self.config.snapshot_every_epochs == 0)

            start_time = time.time()
            print(f"Epoch {epoch}/{until_epoch}")

            # reset performance metrics, as we want to track them per epoch
            if is_snapshot_epoch:
                performance_tracker.reset()

            # Wrap dataloader with tqdm to show a progress bar, unless --no-progress-bar was specified
            iterable_dataloader = self.dataloader
            if not no_progress_bar:
                iterable_dataloader = tqdm(self.dataloader)

            for idx, batch in enumerate(iterable_dataloader, start=1):
                # Transfer data to the appropriate device
                question_embeddings = batch["question_embedding"].to(self.device)
                image_imbeddings = batch["image_embedding"].to(self.device)
                labels = batch["label"].to(self.device)

                # Clear the gradients
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(image_imbeddings, question_embeddings)
                loss = self.loss_function(logits, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Use PerformanceTracker to track the model's accuracy, loss etc
                if is_snapshot_epoch:
                    performance_tracker.update_metrics(logits, labels, loss.item())

            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch} completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, took {elapsed_time/60:.2f} minutes.")

            # In a snapshot-epoch, do post-epoch storage of snapshot and metrics
            if is_snapshot_epoch:
                # Report the performance metrics
                performance_tracker.print_report()
                self.metrics_manager.store_performance_metrics(self.config.model_name, self.dataset_type, epoch, performance_tracker.get_metrics())
                # Save a snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_name = f"snapshot_{self.config.model_name}_{self.dataset_type}_epoch_{epoch}_{timestamp}"
                print(f"Saving snapshot '{snapshot_name}'")

                # Save the model and dataset state
                if self.isModelParallel:
                    # When saving a parallel model, the original model is wrapped and stored in model.module.
                    model_to_save = self.model.module
                else:
                    model_to_save = self.model

                self.snapshot_manager.save_snapshot(snapshot_name, model_to_save, self.optimizer, self.dataset, epoch, performance_tracker.get_metrics()['loss'], lightweight=lightweight_snapshots, skipS3Storage=skip_s3_storage)