import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import os

from tqdm import tqdm
from src.data.vqa_dataset import VQADataset
from src.models.vqa_model import VQAModel
from torch.nn.functional import cross_entropy

def train_model():
    num_workers = int(os.getenv('VQA_NUM_DATALOADER_WORKERS', 1))
    dataset_type = "train"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = int(os.getenv('VQA_NUM_EPOCHS', 5))

    print(f"Torch device: {device}")
    print(f"Using {num_workers} DataLoader workers")

    # load dataset
    dataset = VQADataset(dataset_type)

    # load model
    model = VQAModel(len(dataset.answer_classes))

    # If there's a GPU available...
    if torch.cuda.is_available():
        print('Training on GPU...')
        # Tell PyTorch to use the GPU.
        model = model.to(device)
        # If multiple GPUs are available, wrap model with DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
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
    data_loader = DataLoader(dataset, batch_size=16, num_workers=num_workers, shuffle=True)

    optimizer = Adam(model.parameters())

    print("Beginning training.")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        model.train()  # set model to training mode

        for batch in tqdm(data_loader):
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

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")

# Save model weights
torch.save(model.module.state_dict(), 'vqa_model_weights.pth')

if __name__ == "__main__":
    train_model()
