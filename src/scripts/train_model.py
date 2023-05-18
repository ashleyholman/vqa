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
    num_epochs = int(os.getenv('VQA_NUM_EPOCHS', 5))
    isModelParallel = False

    # Create the snapshot manager
    snapshot_manager = VQASnapshotManager()

    print(f"Torch device: {device}")
    print(f"Using {num_workers} DataLoader workers")

    if args.from_snapshot:
        snapshot = snapshot_manager.load_snapshot(args.from_snapshot, dataset_type)
        if snapshot is None:
            print(f"Snapshot '{args.from_snapshot}' not found.")
            return
        print(f"Using snapshot: {args.from_snapshot}")
        dataset = snapshot.get_dataset()
        model = snapshot.get_model()
        start_epoch = snapshot.get_metadata()["epoch"]
    else:
        # load dataset
        dataset = VQADataset(dataset_type)

        # load model
        model = VQAModel(len(dataset.answer_classes))

        start_epoch = 0

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

    optimizer = Adam(model.parameters())

    print("Beginning training.")
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
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
        print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")

        # Save a snapshot after each epoch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"snapshot_{model.MODEL_NAME}_{timestamp}_epoch_{epoch + 1}"
        print(f"Saving snapshot '{snapshot_name}'")

        # Save the model and dataset state
        if isModelParallel:
            # When saving a parallel model, the original model is wrapped and stored in model.module.
            snapshot_manager.save_snapshot(snapshot_name, model.module, dataset, epoch, lightweight=False)
        else:
            snapshot_manager.save_snapshot(snapshot_name, model, dataset, epoch, lightweight=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VQA model')
    parser.add_argument('--num-dataloader-workers', type=int, default=1, help='Number of dataloader workers')
    parser.add_argument('--dataset-type', type=str, default='train', help='Dataset type to train on (train, validation, mini)')
    parser.add_argument('--num-epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--from-snapshot', type=str, help="Snapshot name to load the model and dataset from.")
    args = parser.parse_args()
    train_model(args)