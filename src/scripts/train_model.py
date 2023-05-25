import argparse
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

# source name for metrics that we emit
METRICS_SOURCE = "train_model"

def train_model(args):
    num_workers = args.num_dataloader_workers
    dataset_type = args.dataset_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = args.num_epochs
    isModelParallel = False

    snapshot_manager = VQASnapshotManager()
    metrics_manager = MetricsManager(METRICS_SOURCE)
    performance_tracker = PerformanceTracker()

    print(f"Torch device: {device}")
    print(f"Using {num_workers} DataLoader workers")

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

        if (snapshot.get_metadata()["model_version"] != VQAModel.MODEL_NAME):
            print(f"ERROR: Model name '{VQAModel.MODEL_NAME}' does not match the model name in the snapshot '{snapshot.get_metadata()['model_name']}'.")
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
        optimizer = Adam(model.parameters(), lr=1e-3)

        # epoch's are 1-indexed for ease of understanding by the user
        start_epoch = 1

    # If there's a GPU available...
    if torch.cuda.is_available():
        print('Training on GPU...')
        # Tell PyTorch to use the GPU.
        model = model.to(device)
        # FIXME: Make this handled in snapshot manager or VQAModel class
        model.answer_embeddings = model.answer_embeddings.to(device)
        # If multiple GPUs are available, wrap model with DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
            isModelParallel = True
    else:
        print('Training on CPU...')

    # Create a DataLoader to handle batching of the dataset
    print("Loading dataset..")
    batch_size = 16
    if isModelParallel:
        batch_size = batch_size * torch.cuda.device_count()
    print(f"Using batch size: {batch_size}")
    data_loader = DataLoader(dataset, batch_size=16, num_workers=num_workers, shuffle=True)

    ### Use class weighting to counteract class imbalance
    class_counts = torch.Tensor(dataset.class_counts)
    # if any classes have 0 count, set them to 1 to avoid dividing by 0
    class_counts[class_counts == 0] = 1
    # take the reciprocal of the class counts so that the most frequent class has the lowest weight
    class_weights = 1. / class_counts
    # normalize weights so they sum to 1
    class_weights = class_weights / class_weights.sum() 
    class_weights = class_weights.to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    print("Beginning training.")
    if (args.skip_s3_storage):
        print("WARNING: Skipping S3 storage of snapshots.  Snapshots will only be stored locally.")

    for epoch in range(start_epoch, num_epochs+1):
        start_time = time.time()
        print(f"Epoch {epoch}/{num_epochs}")

        # reset performance metrics, as we want to track them per epoch
        performance_tracker.reset()

        model.train()  # set model to training mode

        # Wrap data_loader with tqdm to show a progress bar, unless --no-progress-bar was specified
        iterable_data_loader = data_loader
        if not args.no_progress_bar:
            iterable_data_loader = tqdm(data_loader)

        for idx, batch in enumerate(iterable_data_loader, start=1):
            # Transfer data to the appropriate device
            question_embeddings = batch["question_embedding"].to(device)
            image_imbeddings = batch["image_embedding"].to(device)
            labels = batch["label"].to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(image_imbeddings, question_embeddings)
            loss = loss_function(logits, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Use PerformanceTracker to track the model's accuracy, loss etc
            performance_tracker.update_metrics(logits, labels)

            # Print average loss every 500 batches
            if idx % 500 == 0:
                print(f"\nEpoch {epoch}, Batch {idx}, Average Loss: {performance_tracker.get_metrics()['loss']:.4f}")

        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch} completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, took {elapsed_time/60:.2f} minutes.")


        # Report the performance metrics
        performance_tracker.print_report()

        # Store the metrics 
        metrics_manager.store_performance_metrics(model.MODEL_NAME, dataset_type, epoch, performance_tracker.get_metrics())

        # Save a snapshot after each epoch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"snapshot_{model.MODEL_NAME}_{dataset_type}_epoch_{epoch}_{timestamp}"
        print(f"Saving snapshot '{snapshot_name}'")

        # Save the model and dataset state
        if isModelParallel:
            # When saving a parallel model, the original model is wrapped and stored in model.module.
            snapshot_manager.save_snapshot(snapshot_name, model.module, optimizer, dataset, epoch, performance_tracker.get_metrics()['loss'], lightweight=args.lightweight_snapshots, skipS3Storage=args.skip_s3_storage)
        else:
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