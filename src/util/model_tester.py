import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from torch.nn.functional import cross_entropy

from src.metrics.error_tracker import ErrorTracker
from src.metrics.performance_tracker import PerformanceTracker

class ModelTester:
    def __init__(self, config, dataset, num_dataloader_workers):
        self.config = config
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a DataLoader to handle batching of the dataset
        self.dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=num_dataloader_workers, shuffle=config.shuffle)

    def test(self, model, performance_tracker: PerformanceTracker, error_tracker: ErrorTracker, device, no_progress_bar=False):
        # Move model to evaluation mode
        model.eval()

        # Wrap data_loader with tqdm to show a progress bar, unless --no-progress-bar was specified
        dataloader = self.dataloader
        if not no_progress_bar:
            total_batches = len(dataloader)
            if self.config.max_batches_per_epoch:
                total_batches = min(self.config.max_batches_per_epoch, total_batches)
            dataloader = tqdm(dataloader, total=total_batches)

        # No need to track gradients for this
        with torch.no_grad():
            for idx, batch in enumerate(dataloader, start=1):
                # Transfer data to the appropriate device
                if self.config.finetune_from_snapshot:
                    batch["image"] = batch["image"].to(self.device)
                    batch["input_ids"] = batch["input_ids"].to(self.device)
                    batch["attention_mask"] = batch["attention_mask"].to(self.device)
                else:
                    batch["image_embedding"] = batch["image_embedding"].to(self.device)
                    batch["question_embedding"] = batch["question_embedding"].to(self.device)

                labels = batch["label"].to(device)

                # Run the model and get the predictions
                logits = model(batch)

                # Calculate the loss
                loss = cross_entropy(logits, labels)

                # Update the performance tracker
                if performance_tracker:
                    performance_tracker.update_metrics(logits, labels, loss.item())

                # Update error tracker
                if error_tracker:
                    error_tracker.update_instance_data(logits, labels, batch["question_id"])

                if idx % 500 == 0:
                    print(f"\nBatch {idx}..")

                # Limit the number of batches per epoch if configured to do so
                if self.config.max_batches_per_epoch and idx >= self.config.max_batches_per_epoch:
                    print(f"Reached max batches per epoch ({self.config.max_batches_per_epoch}).")
                    break