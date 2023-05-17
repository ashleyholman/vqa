import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
import os

from src.data.vqa_dataset import VQADataset
from src.models.vqa_model import VQAModel

def top_k_correct(output, target, k):
    """Computes the count of correct predictions in the top k outputs."""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:k].reshape(-1).float().sum()
        return correct_k

def main():
    num_workers = int(os.getenv('VQA_NUM_DATALOADER_WORKERS', 1))
    dataset_type = os.getenv('VQA_DATASET_TYPE', "mini")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Torch device: {device}")
    print(f"Using {num_workers} DataLoader workers")

    # load dataset
    dataset = VQADataset(dataset_type)

    # Store the model's predictions and the correct answers here
    predictions = []
    correct_answers = []
    top_5_correct = 0

    # load model
    model = VQAModel(len(dataset.answer_classes))

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
            #if top_5_correct_in_batch > 0:
            #  print(f"\nTop 5 correct in batch: {top_5_correct_in_batch}")
            top_5_correct += top_5_correct_in_batch

    # At this point, predictions and correct_answers are lists containing the model's
    # predictions and the correct answers, respectively.
    accuracy = sum(p == ca for p, ca in zip(predictions, correct_answers)) / len(predictions) * 100
    top_5_acc = (top_5_correct / len(predictions)) * 100
    print(f"\nUntrained model accuracy: {accuracy:.2f}%")
    print(f'Top-5 Accuracy: {top_5_acc:.2f}%')

if __name__ == "__main__":
    main()