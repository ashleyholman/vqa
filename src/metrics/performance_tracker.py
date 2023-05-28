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

import numpy as np

class GiniCoefficientMetric(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []

    def update(self, logits, labels):
        _, preds = torch.max(logits, dim=1)
        self.predictions.extend(preds.tolist())

    def value(self):
        return self._gini_coefficient(self.predictions)

    @staticmethod
    def _gini_coefficient(predictions):
        # convert to numpy array for calculation
        predictions = np.array(predictions)

        # sort the predictions
        predictions_sorted = np.sort(predictions)

        # if all predictions are the same, return 1
        if np.unique(predictions_sorted).size == 1:
            return 1.0

        # calculate the Gini coefficient
        n = len(predictions_sorted)
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n  - 1) * predictions_sorted)) / (n * np.sum(predictions_sorted))

        return gini

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
            "f1_score_macro": F1Metric("macro"),
            "gini_coefficient": GiniCoefficientMetric()
        }
        self.cached_metric_values = None
        self.reset()

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def update_metrics(self, logits, labels, loss=None):
        self.cached_metric_values = None

        for metric_name, metric in self.metrics.items():
            if metric_name == 'loss':
                metric.update(loss, logits.size(0))
            else:
                metric.update(logits, labels)

    def get_metrics(self):
        # calculating the metrics is expensive on a large dataset, and this
        # function tends to get called repeatedly after an epoch finishes, so we
        # cache the results here
        if self.cached_metric_values is None:
            self.cached_metric_values = {name: metric.value() for name, metric in self.metrics.items()}

        return self.cached_metric_values

    def print_report(self):
        print("== Performance Metrics ==")
        for metric_name, metric in self.metrics.items():
            print(f"{metric_name}: {metric.value():.4f}")
