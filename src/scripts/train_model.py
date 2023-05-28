import argparse
import time
import torch

from datetime import datetime
from torch.optim import Adam

from src.models.model_configuration import ModelConfiguration
from src.models.vqa_model import VQAModel
from src.util.model_trainer import ModelTrainer
from src.snapshots.vqa_snapshot_manager import VQASnapshotManager
from src.metrics.metrics_manager import MetricsManager
from src.metrics.performance_tracker import PerformanceTracker
from src.data.vqa_dataset import VQADataset

METRICS_SOURCE = "train_model"

def train_model(args):
    dataset_type = args.dataset_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_dataloader_workers = args.num_dataloader_workers
    num_epochs = args.num_epochs

    config = ModelConfiguration()
    snapshot_manager = VQASnapshotManager()
    metrics_manager = MetricsManager(METRICS_SOURCE)

    if args.from_snapshot:
        try:
            snapshot = snapshot_manager.load_snapshot(args.from_snapshot, dataset_type, device)
        except Exception as e:
            print(f"Failed to load snapshot: {e}")
            return

        # Since we're resuming training, the dataset and model version we're training
        # on should be the same as the ones in the snapshot, otherwise we'll get
        # unexpected results.
        if (snapshot.get_metadata()["settype"] != dataset_type):
            print(f"ERROR: Dataset type '{dataset_type}' does not match the dataset type in the snapshot '{snapshot.get_metadata()['dataset_type']}'.")
            return

        if (snapshot.get_metadata()["model_version"] != config.model_name):
            print(f"ERROR: Model name '{config.model_name}' does not match the model name in the snapshot '{snapshot.get_metadata()['model_name']}'.")
            return

        print(f"Using snapshot: {args.from_snapshot}")
        dataset = snapshot.get_dataset()
        model = snapshot.get_model()
        optimizer = snapshot.get_optimizer()

        # The epoch number stored in the snapshot represents
        # the last completed training epoch, so resume training from epoch+1
        start_epoch = snapshot.get_metadata()["epoch"]+1
        if (start_epoch > num_epochs):
            print(f"Snapshot '{args.from_snapshot}' already trained for {start_epoch-1} epochs.  Nothing to do.")
            return

        if (snapshot.isLightweight()):
            print(f"WARNING: Snapshot '{args.from_snapshot}' is lightweight, meaning the pretrained "
                  "weights of the BERT and ViT models are not included in the snapshot.  Those models "
                  "will be initialized from huggingface.  If the underlying weights of these models "
                  "have changed since the snapshot was created, the model may not train correctly.")

        print(f"Resuming training from epoch {start_epoch}.")
    else:
        # load dataset
        dataset = VQADataset(dataset_type)

        # load model
        print("Answer classes: ", dataset.answer_classes)
        model = VQAModel(dataset.answer_classes)

        # Create a new optimizer
        optimizer = Adam(model.parameters(), lr=config.learning_rate)

        # epoch's are 1-indexed for ease of understanding by the user
        start_epoch = 1

    model_trainer = ModelTrainer(config, model, dataset, optimizer, num_dataloader_workers)
    performance_tracker = PerformanceTracker()

    for epoch in range(start_epoch, args.num_epochs+1):
        print(f"Epoch {epoch}/{args.num_epochs}")

        is_snapshot_epoch = (epoch % config.snapshot_every_epochs == 0)
        is_validation_epoch = True

        start_time = time.time()

        if is_validation_epoch:
            performance_tracker.reset()
            model_trainer.train_one_epoch(performance_tracker)
        else:
            # Train without performance tracking, which is faster.
            model_trainer.train_one_epoch()

        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch} completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, took {elapsed_time/60:.2f} minutes.")

        if is_validation_epoch:
            performance_tracker.print_report()
            metrics_manager.store_performance_metrics(config.model_name, dataset_type, epoch, performance_tracker.get_metrics())

        if is_snapshot_epoch:
            # Save a snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_name = f"snapshot_{config.model_name}_{dataset_type}_epoch_{epoch}_{timestamp}"
            print(f"Saving snapshot '{snapshot_name}'")

            # Save the model and dataset state
            snapshot_manager.save_snapshot(snapshot_name, model, optimizer, dataset, epoch, performance_tracker.get_metrics()['loss'], lightweight=args.lightweight_snapshots, skipS3Storage=args.skip_s3_storage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VQA model')
    parser.add_argument('--num-dataloader-workers', type=int, default=1, help='Number of dataloader workers')
    parser.add_argument('--dataset-type', type=str, default='train', help='Dataset type to train on (train, validation, mini)')
    parser.add_argument('--num-epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--from-snapshot', type=str, help="Snapshot name to load the model and dataset from.")
    parser.add_argument('--lightweight-snapshots', action='store_true', help="Use this flag to save lightweight snapshots only (doesn't save pretrained bert or vit weights)")
    parser.add_argument('--skip-s3-storage', action='store_true', help='Use this flag to skip storing new snapshots in S3')
    parser.add_argument('--no-progress-bar', action='store_true', help='Use this flag to disable the progress bar during training')

    args = parser.parse_args()
    train_model(args)