import argparse
import torch

from src.data.vqa_dataset import VQADataset
from src.models.vqa_model import VQAModel
from src.snapshots.vqa_snapshot_manager import VQASnapshotManager
from src.metrics.metrics_manager import MetricsManager
from src.metrics.performance_tracker import PerformanceTracker
from src.models.model_configuration import ModelConfiguration
from src.util.model_tester import ModelTester

# source name for metrics that we emit
METRICS_SOURCE = "test_model"

def main(args):
    num_workers = args.num_dataloader_workers
    dataset_type = args.dataset_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Torch device: {device}")
    print(f"Using {num_workers} DataLoader workers")
    print(f"Using dataset type: {dataset_type}")

    snapshot_manager = VQASnapshotManager()
    metrics_manager = MetricsManager(METRICS_SOURCE)
    performance_tracker = PerformanceTracker()
    config = ModelConfiguration()

    model_name = None;

    if args.from_snapshot:
        snapshot = snapshot_manager.load_snapshot(args.from_snapshot, dataset_type, device)
        if snapshot is None:
            print(f"Snapshot '{args.from_snapshot}' not found.")
            return
        model = snapshot.get_model()
        dataset = snapshot.get_dataset()
        model_name = snapshot.get_metadata().get("model_version")
        print(f"Using snapshot: {args.from_snapshot}")
    else:
        # instantiate dataset and model from scratch
        dataset = VQADataset(dataset_type)
        model = VQAModel(dataset.answer_classes)
        model_name = config.model_name

        print("Using untrained model")

    model.to(device)

    if config.use_answer_embeddings:
    # FIXME: Handle this in snapshot manager or VQADataset class
      model.answer_embeddings = model.answer_embeddings.to(device)

    print("Initialising ModleTester..")
    model_tester = ModelTester(config, dataset, num_workers)

    print("Evaluating model...")
    model_tester.test(model, performance_tracker, device, args.no_progress_bar)

    # Print performance report
    performance_tracker.print_report()

    # Determine what epoch number this model has been trained to, for storing performance metrics.
    # If this model wasn't loaded from a snapshot, it means it was an untrained model (epoch 0)
    if args.from_snapshot:
        epoch = snapshot.get_metadata()['epoch']
    else:
        epoch = 0

    # Store the metrics to DynamoDB for later reporting
    metrics_manager.store_performance_metrics(model_name, dataset_type, epoch, performance_tracker.get_metrics())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-snapshot', type=str, help="Snapshot name to load the model and dataset from.")
    parser.add_argument('--dataset-type', type=str, default="validation", help="Dataset type to use (train, validation, mini, etc.)")
    parser.add_argument('--num-dataloader-workers', type=int, default=1, help="Number of dataloader workers to use.")
    parser.add_argument('--no-progress-bar', action='store_true', help="Do not display progress bar during execution.")

    args = parser.parse_args()
    main(args)
