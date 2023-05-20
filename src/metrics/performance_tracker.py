from typing import Any
import torch
from torch.nn.functional import cross_entropy
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    source: str
    dataset_type: str
    accuracy: float
    top_5_accuracy: float
    loss: float

    def print_report(self):
        print("== Performance Metrics ==")
        print(f"Accuracy: {self.accuracy:.4f}")
        print(f"Top 5 Accuracy: {self.top_5_accuracy:.4f}")
        print(f"Loss: {self.loss:.4f}")

class PerformanceTracker:
    def __init__(self, source, dataset_type):
        self.source = source
        self.dataset_type = dataset_type

        self.reset()
    
    def reset(self):
        self.predictions = []
        self.correct_answers = []
        self.top_5_correct = 0
        self.total_loss = 0.0

    def update_metrics(self, logits, labels):
        _, preds = torch.max(logits, dim=1)
        
        loss = cross_entropy(logits, labels)
        self.total_loss += loss.item() * logits.size(0)  # multiply loss value by batch size

        self.predictions.extend(preds.tolist())
        self.correct_answers.extend(labels.tolist())
        self.top_5_correct += self._top_k_correct(logits, labels, 5)

    @staticmethod
    def _top_k_correct(logits, labels, k):
        _, top_k_preds = logits.topk(k, dim=1)
        labels_repeated = labels.view(-1, 1).expand_as(top_k_preds)
        correct = top_k_preds.eq(labels_repeated).sum()
        return correct.item()

    def get_metrics(self):
        accuracy = sum(p == ca for p, ca in zip(self.predictions, self.correct_answers)) / len(self.predictions) * 100
        top_5_acc = (self.top_5_correct / len(self.predictions)) * 100
        avg_loss = self.total_loss / len(self.predictions)  # calculate average loss per sample

        return PerformanceMetrics(self.source, self.dataset_type, accuracy=accuracy, top_5_accuracy=top_5_acc, loss=avg_loss)