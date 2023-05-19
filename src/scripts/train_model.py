import argparse
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from datetime import datetime

from tqdm import tqdm
from src.data.vqa_dataset import VQADataset
from src.models.vqa_model import VQAModel
from torch.nn.functional import cross_entropy

from src.snapshots.vqa_snapshot_manager import VQASnapshotManager
from src.snapshots.snapshot import Snapshot

def train_model(args):
    num_workers = args.num_dataloader_workers
    dataset_type = args.dataset_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = args.num_epochs
    isModelParallel = False

    # Create the snapshot manager
    snapshot_manager = VQASnapshotManager()

    print(f"Torch device: {device}")
    print(f"Using {num_workers} DataLoader workers")

    if args.from_snapshot:
        try:
            snapshot = snapshot_manager.load_snapshot(args.from_snapshot, dataset_type)
        except Exception as e:
            print(f"Failed to load snapshot: {e}")
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
        model = VQAModel(len(dataset.answer_classes))

        # Create a new optimizer
        optimizer = Adam(model.parameters())

        # epoch's are 1-indexed for ease of understanding by the user
        start_epoch = 1

    # If there's a GPU available...
    if torch.cuda.is_available():
        print('Training on GPU...')
        # Tell PyTorch to use the GPU.
        model = model.to(device)
        # If multiple GPUs are available, wrap model with DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
            isModelParallel = True
    else:
        print('Training on CPU...')

    # We want to freeze the BERT and ViT parameters for now.  Just train our model's new layers.
    print(f"Freezing BERT and ViT parameters")
    for param in model.vit.parameters():
        param.requires_grad = False

    for param in model.bert.parameters():
        param.requires_grad = False

    # Create a DataLoader to handle batching of the dataset
    print("Loading dataset..")
    batch_size = 16
    if isModelParallel:
        batch_size = batch_size * torch.cuda.device_count()
    print(f"Using batch size: {batch_size}")
    data_loader = DataLoader(dataset, batch_size=16, num_workers=num_workers, shuffle=True)

    print("Beginning training.")
    if (args.skip_s3_storage):
        print("WARNING: Skipping S3 storage of snapshots.  Snapshots will only be stored locally.")

    for epoch in range(start_epoch, num_epochs+1):
        print(f"Epoch {epoch}/{num_epochs}")
        running_loss = 0.0
        model.train()  # set model to training mode

        for idx, batch in enumerate(tqdm(data_loader), start=1):
            # Transfer data to the appropriate device
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(images, input_ids, attention_mask)
            loss = cross_entropy(logits, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Print average loss every 500 batches
            if idx % 500 == 0:
                print(f"\nBatch {idx}, Average Loss: {running_loss / (idx * images.size(0)):.4f}")

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch} loss: {epoch_loss:.4f}")

        # Save a snapshot after each epoch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"snapshot_{model.MODEL_NAME}_{dataset_type}_epoch_{epoch}_{timestamp}"
        print(f"Saving snapshot '{snapshot_name}'")

        # Save the model and dataset state
        if isModelParallel:
            # When saving a parallel model, the original model is wrapped and stored in model.module.
            snapshot_manager.save_snapshot(snapshot_name, model.module, optimizer, dataset, epoch, loss, lightweight=args.lightweight_snapshots, skipS3Storage=args.skip_s3_storage)
        else:
            snapshot_manager.save_snapshot(snapshot_name, model, optimizer, dataset, epoch, loss, lightweight=args.lightweight_snapshots, skipS3Storage=args.skip_s3_storage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VQA model')
    parser.add_argument('--num-dataloader-workers', type=int, default=1, help='Number of dataloader workers')
    parser.add_argument('--dataset-type', type=str, default='train', help='Dataset type to train on (train, validation, mini)')
    parser.add_argument('--num-epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--from-snapshot', type=str, help="Snapshot name to load the model and dataset from.")
    parser.add_argument('--lightweight-snapshots', action='store_true', help="Use this flag to save lightweight snapshots only (doesn't save pretrained bert or vit weights)")
    parser.add_argument('--skip-s3-storage', action='store_true', help='Use this flag to skip storing new snapshots in S3')
    args = parser.parse_args()
    train_model(args)