import argparse
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
import os

from src.data.vqa_dataset import VQADataset
from src.models.vqa_model import VQAModel
from src.snapshots.vqa_snapshot_manager import VQASnapshotManager
from src.metrics.metrics_manager import MetricsManager

def top_k_correct(output, target, k):
    """Computes the count of correct predictions in the top k outputs."""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:k].reshape(-1).float().sum()
        return correct_k

def main(args):
    num_workers = args.num_dataloader_workers
    dataset_type = args.dataset_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Torch device: {device}")
    print(f"Using {num_workers} DataLoader workers")
    print(f"Using dataset type: {dataset_type}")

    snapshot_manager = VQASnapshotManager()
    metrics_manager = MetricsManager()

    if args.from_snapshot:
        snapshot = snapshot_manager.load_snapshot(args.from_snapshot, dataset_type)
        if snapshot is None:
            print(f"Snapshot '{args.from_snapshot}' not found.")
            return
        model = snapshot.get_model()
        dataset = snapshot.get_dataset()
        print(f"Using snapshot: {args.from_snapshot}")
    else:
        # instantiate dataset and model from scratch
        dataset = VQADataset(dataset_type)
        model = VQAModel(len(dataset.answer_classes))

        print("Using untrained model")

    model.to(device)

    # Store the model's predictions and the correct answers here
    predictions = []
    correct_answers = []
    top_5_correct = 0

    # Create a DataLoader to handle batching of the dataset
    data_loader = DataLoader(dataset, batch_size=16, num_workers=num_workers, shuffle=False)

    # Move model to evaluation mode
    model.eval()

    # No need to track gradients for this
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # Transfer data to the appropriate device
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Run the model and get the predictions
            logits = model(images, input_ids, attention_mask)
            _, preds = torch.max(logits, dim=1)

            # Append preds and labels to lists
            predictions.extend(preds.tolist())
            correct_answers.extend(labels.tolist())
            top_5_correct_in_batch = top_k_correct(logits, labels, 5)
            top_5_correct += top_5_correct_in_batch

    # At this point, predictions and correct_answers are lists containing the model's
    # predictions and the correct answers, respectively.
    accuracy = sum(p == ca for p, ca in zip(predictions, correct_answers)) / len(predictions) * 100
    top_5_acc = (top_5_correct / len(predictions)) * 100
    print(f"\nModel accuracy: {accuracy:.2f}%")
    print(f'Top-5 Accuracy: {top_5_acc:.2f}%')

    # Determine what epoch number this model has been trained to, for storing performance metrics.
    # If this model wasn't loaded from a snapshot, it means it was an untrained model (epoch 0)
    if args.from_snapshot:
        epoch = snapshot.get_metadata()['epoch']
    else:
        epoch = 0

    # Store the metrics to DynamoDB for later reporting
    metrics_manager.store_performance_metrics(model.MODEL_NAME, dataset_type, epoch, accuracy, top_5_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-snapshot', type=str, help="Snapshot name to load the model and dataset from.")
    parser.add_argument('--dataset-type', type=str, default="validation", help="Dataset type to use (train, validation, mini, etc.)")
    parser.add_argument('--num-dataloader-workers', type=int, default=1, help="Number of dataloader workers to use.")
    args = parser.parse_args()
    main(args)
