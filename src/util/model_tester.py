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

        # Create a DataLoader to handle batching of the dataset
        self.dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=num_dataloader_workers, shuffle=config.shuffle)

    def test(self, model, performance_tracker: PerformanceTracker, error_tracker: ErrorTracker, device, no_progress_bar=False):
        # Move model to evaluation mode
        model.eval()

        # Wrap data_loader with tqdm to show a progress bar, unless --no-progress-bar was specified
        dataloader = self.dataloader
        if not no_progress_bar:
            dataloader = tqdm(dataloader)

        # No need to track gradients for this
        with torch.no_grad():
            for idx, batch in enumerate(dataloader, start=1):
                # Transfer data to the appropriate device
                question_embeddings = batch["question_embedding"].to(device)
                image_embeddings = batch["image_embedding"].to(device)
                labels = batch["label"].to(device)

                # Run the model and get the predictions
                logits = model(image_embeddings, question_embeddings)

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