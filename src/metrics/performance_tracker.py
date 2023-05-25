from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import torch
from torch.nn.functional import cross_entropy
from sklearn.metrics import precision_score, recall_score, f1_score

class Metric(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, logits, labels):
        pass

    @abstractmethod
    def value(self):
        pass

class LossMetric(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_count = 0

    def update(self, loss_value, count):
        self.total_loss += loss_value * count
        self.total_count += count

    def value(self):
        return self.total_loss / max(1, self.total_count)

class AccuracyMetric(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.correct_answers = []

    def update(self, logits, labels):
        _, preds = torch.max(logits, dim=1)
        self.predictions.extend(preds.tolist())
        self.correct_answers.extend(labels.tolist())

    def value(self):
        return sum(p == ca for p, ca in zip(self.predictions, self.correct_answers)) / max(1, len(self.predictions)) * 100

class TopKAccuracyMetric(Metric):
    def __init__(self, k):
        self.k = k
        self.reset()

    def reset(self):
        self.top_k_correct = 0
        self.total_count = 0

    def update(self, logits, labels):
        self.top_k_correct += self._top_k_correct(logits, labels, self.k)
        self.total_count += logits.size(0)

    def value(self):
        return self.top_k_correct / max(1, self.total_count) * 100

    @staticmethod
    def _top_k_correct(logits, labels, k):
        _, top_k_preds = logits.topk(k, dim=1)
        labels_repeated = labels.view(-1, 1).expand_as(top_k_preds)
        correct = top_k_preds.eq(labels_repeated).sum()
        return correct.item()

class PrecisionMetric(Metric):
    def __init__(self, average):
        self.average = average
        self.reset()

    def reset(self):
        self.predictions = []
        self.correct_answers = []

    def update(self, logits, labels):
        _, preds = torch.max(logits, dim=1)
        self.predictions.extend(preds.tolist())
        self.correct_answers.extend(labels.tolist())

    def value(self):
        return precision_score(self.correct_answers, self.predictions, average=self.average, zero_division=1) * 100

class RecallMetric(Metric):
    def __init__(self, average):
        self.average = average
        self.reset()

    def reset(self):
        self.predictions = []
        self.correct_answers = []

    def update(self, logits, labels):
        _, preds = torch.max(logits, dim=1)
        self.predictions.extend(preds.tolist())
        self.correct_answers.extend(labels.tolist())

    def value(self):
        return recall_score(self.correct_answers, self.predictions, average=self.average, zero_division=1) * 100

class F1Metric(Metric):
    def __init__(self, average):
        self.average = average
        self.reset()

    def reset(self):
        self.predictions = []
        self.correct_answers = []

    def update(self, logits, labels):
        _, preds = torch.max(logits, dim=1)
        self.predictions.extend(preds.tolist())
        self.correct_answers.extend(labels.tolist())

    def value(self):
        return f1_score(self.correct_answers, self.predictions, average=self.average, zero_division=1) * 100

class PerformanceTracker:
    def __init__(self):
        self.metrics = {
            "accuracy": AccuracyMetric(),
            "top_5_accuracy": TopKAccuracyMetric(5),
            "loss": LossMetric(),
            "precision_micro": PrecisionMetric("micro"),
            "precision_macro": PrecisionMetric("macro"),
            "recall_micro": RecallMetric("micro"),
            "recall_macro": RecallMetric("macro"),
            "f1_score_micro": F1Metric("micro"),
            "f1_score_macro": F1Metric("macro")
        }
        self.reset()

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def update_metrics(self, logits, labels, loss):
        for metric in self.metrics:
            if isinstance(metric, LossMetric):
                # special case for LossMetric
                metric.update(loss, logits.size(0))
            else:
                metric.update(logits, labels)
    def update_metrics(self, logits, labels, loss=None):
        for metric_name, metric in self.metrics.items():
            if metric_name == 'loss':
                metric.update(loss, logits.size(0))
            else:
                metric.update(logits, labels)

    def get_metrics(self):
        return {name: metric.value() for name, metric in self.metrics.items()}

    def print_report(self):
        print("== Performance Metrics ==")
        for metric_name, metric in self.metrics.items():
            print(f"{metric_name}: {metric.value():.4f}")
