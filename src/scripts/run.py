import argparse
import datetime
import time
import torch

from torch.optim import Adam
from datetime import datetime


from src.data.vqa_dataset import VQADataset
from src.metrics.metrics_manager import MetricsManager
from src.metrics.performance_tracker import PerformanceTracker

from src.models.model_configuration import ModelConfiguration
from src.models.vqa_model import VQAModel
from src.snapshots.vqa_snapshot_manager import VQASnapshotManager
from src.util.model_tester import ModelTester
from src.util.model_trainer import ModelTrainer

TRAINING_METRICS_SOURCE = "train_model"
VALIDATION_METRICS_SOURCE = "test_model"

def get_restore_point(training_dataset_type, device):
    snapshot_manager = VQASnapshotManager()

    # TODO: implement discovery of snapshot
    '''
    try:
        snapshot = snapshot_manager.load_snapshot(args.from_snapshot, training_dataset_type, device)
    except Exception as e:
        print(f"Failed to load snapshot: {e}")
        return
    '''
    return None


def run(config, args):
    '''Run the training and validation process.'''
    if args.use_mini_dataset:
        training_dataset_type = 'mini'
        validation_dataset_type = 'mini'
    else:
        training_dataset_type = 'train'
        validation_dataset_type = 'validation'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_dataloader_workers = args.num_dataloader_workers
    max_epochs = args.max_epochs

    training_metrics_manager = MetricsManager(TRAINING_METRICS_SOURCE)
    validation_metrics_manager = MetricsManager(VALIDATION_METRICS_SOURCE)
    snapshot_manager = VQASnapshotManager()

    # TODO: auto-detect resumption instead of args.from_snapshot here.
    snapshot = get_restore_point(training_dataset_type, device)
    if snapshot:
        # Since we're resuming training, the dataset and model version we're training
        # on should be the same as the ones in the snapshot, otherwise we'll get
        # unexpected results.
        if (snapshot.get_metadata()["settype"] != training_dataset_type):
            print(f"ERROR: Dataset type '{training_dataset_type}' does not match the dataset type in the snapshot '{snapshot.get_metadata()['dataset_type']}'.")
            return

        if (snapshot.get_metadata()["model_version"] != config.model_name):
            print(f"ERROR: Model name '{config.model_name}' does not match the model name in the snapshot '{snapshot.get_metadata()['model_name']}'.")
            return

        print(f"Using snapshot: {snapshot.get_name()}")
        training_dataset = snapshot.get_dataset()
        model = snapshot.get_model()
        optimizer = snapshot.get_optimizer()

        # The epoch number stored in the snapshot represents
        # the last completed training epoch, so resume training from epoch+1
        start_epoch = snapshot.get_metadata()["epoch"]+1
        if (start_epoch > max_epochs):
            print(f"Snapshot '{snapshot.get_name()}' already trained for {start_epoch-1} epochs.  Nothing to do.")
            return

        if (snapshot.isLightweight()):
            print(f"WARNING: Snapshot '{snapshot.get_name()}' is lightweight, meaning the pretrained "
                  "weights of the BERT and ViT models are not included in the snapshot.  Those models "
                  "will be initialized from huggingface.  If the underlying weights of these models "
                  "have changed since the snapshot was created, the model may not train correctly.")

        print(f"Resuming training from epoch {start_epoch}.")
    else:
        # load training dataset
        print("Loading training dataset...")
        training_dataset = VQADataset(training_dataset_type)

        # load model
        print("Answer classes: ", training_dataset.answer_classes)
        model = VQAModel(training_dataset.answer_classes)

        # Create a new optimizer
        optimizer = Adam(model.parameters(), lr=config.learning_rate)

        # epoch's are 1-indexed for ease of understanding by the user
        start_epoch = 1
    
    model.to(device)
    if config.use_answer_embeddings:
        # FIXME: Handle this in snapshot manager or VQADataset class
        model.answer_embeddings = model.answer_embeddings.to(device)
    
    # Load the validation dataset, using the same answer classes as the training set.
    print("Loading validation dataset...")
    validation_dataset = VQADataset(validation_dataset_type, training_dataset.answer_classes)

    model_trainer = ModelTrainer(config, model, training_dataset, optimizer, num_dataloader_workers)
    model_tester = ModelTester(config, validation_dataset, num_dataloader_workers)
    performance_tracker_reusable = PerformanceTracker()

    start_epoch = 1

    for epoch in range(start_epoch, args.max_epochs+1):
        print(f"Epoch {epoch}/{args.max_epochs}")

        is_snapshot_epoch = (epoch % config.snapshot_every_epochs == 0)
        is_metrics_epoch = (epoch % config.metrics_every_epochs == 0)
        performance_tracker = None

        start_time = time.time()
        
        if is_metrics_epoch:
            performance_tracker = performance_tracker_reusable
            performance_tracker.reset()
        
        print("Begin training...")
        model_trainer.train_one_epoch(performance_tracker)

        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch} completed training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, took {elapsed_time/60:.2f} minutes.")

        if is_metrics_epoch:
            performance_tracker.print_report()
            training_metrics_manager.store_performance_metrics(config.model_name, training_dataset_type, epoch, performance_tracker.get_metrics())

        if is_snapshot_epoch:
            # Save a snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_name = f"snapshot_{config.model_name}_{training_dataset_type}_epoch_{epoch}_{timestamp}"
            print(f"Saving snapshot '{snapshot_name}'")

            # Save the model and dataset state
            snapshot_manager.save_snapshot(snapshot_name, model, optimizer, training_dataset, epoch, performance_tracker.get_metrics()['loss'], lightweight=False, skipS3Storage=args.skip_s3_storage)
        
        if not is_metrics_epoch:
            # if it's not a metrics epoch, we don't need to do validation.  Continue to the next epoch.
            continue

        print("Validating model...")
        performance_tracker.reset()
        model_tester.test(model, performance_tracker, device, args.no_progress_bar)

        # Print performance report
        performance_tracker.print_report()

        # store metrics
        validation_metrics_manager.store_performance_metrics(config.model_name, validation_dataset_type, epoch, performance_tracker.get_metrics())

    print("Run complete.")

if __name__ == "__main__":
    config = ModelConfiguration()

    parser = argparse.ArgumentParser(description='Execute a training and validation run.')
    parser.add_argument('--num-dataloader-workers', type=int, default=1, help='Number of dataloader workers')
    parser.add_argument('--max-epochs', type=int, default=config.max_epochs, help='The maximum number of epochs to train for.')
    parser.add_argument('--skip-s3-storage', action='store_true', help='Use this flag to skip storing new snapshots in S3')
    parser.add_argument('--no-progress-bar', action='store_true', help='Use this flag to disable the progress bar during training')
    parser.add_argument('--use-mini-dataset', action='store_true', help='For local testing.  Both training and validation are done on the mini dataset.')

    args = parser.parse_args()
    run(config, args)