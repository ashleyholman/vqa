import torch

class ErrorTracker:

    def __init__(self, num_classes, max_samples_per_class=100, track_true_negatives=False, topk=5, topk_as_positive=False):
        self.num_classes = num_classes
        self.max_samples_per_class = max_samples_per_class
        self.track_true_negatives = track_true_negatives
        self.topk = topk
        self.topk_as_positive = topk_as_positive
        self.instance_data_per_class = {
            i: {
                'TP': [],  # True Positives
                'FP': [],  # False Positives
                'FN': [],  # False Negatives
                'TN': [] if track_true_negatives else None  # True Negatives
            } for i in range(num_classes)
        }
        self.error_count_per_class = {
            i: {
                'TP': 0,  # True Positives
                'FP': 0,  # False Positives
                'FN': 0,  # False Negatives
                'TN': 0 if track_true_negatives else None  # True Negatives
            } for i in range(num_classes)
        }

    def _add_to_class_data(self, class_id, error_type, data):
        self.error_count_per_class[class_id][error_type] += 1
        if len(self.instance_data_per_class[class_id][error_type]) < self.max_samples_per_class:
            self.instance_data_per_class[class_id][error_type].append(data)

    def update_instance_data(self, logits, labels, question_ids):
        confidences = torch.softmax(logits, dim=1)
        _, preds = torch.topk(confidences, self.topk, dim=1)

        for i in range(logits.shape[0]):
            # Create array of objects with 'class_id' and 'confidence'
            predicted_classes = [{'class_id': preds[i][j].item(), 'confidence': confidences[i][preds[i][j]].item()} for j in range(self.topk)]
            
            data = {
                'true_class': labels[i].item(),
                'predicted_classes': predicted_classes,  # use the array of objects
                'question_id': question_ids[i].item()
            }

            true_class = labels[i].item()

            # Check if we consider top k predictions as positive or only the top prediction
            positive_class_ids = [class_obj['class_id'] for class_obj in predicted_classes] if self.topk_as_positive else [predicted_classes[0]['class_id']]

            for positive_class_id in positive_class_ids:
                if positive_class_id == true_class:
                    self._add_to_class_data(positive_class_id, 'TP', data)
                else:
                    self._add_to_class_data(positive_class_id, 'FP', data)

            if true_class not in positive_class_ids:
                self._add_to_class_data(true_class, 'FN', data)

            if self.track_true_negatives:
                for class_id in range(self.num_classes):
                    if class_id != true_class and class_id not in positive_class_ids:
                        self._add_to_class_data(class_id, 'TN', data)

    def get_instance_data(self):
        return {
            'sample_questions' : self.instance_data_per_class,
            'counts' : self.error_count_per_class
        }

    def reset(self):
        self.instance_data_per_class = {
            i: {
                'TP': [],  # True Positives
                'FP': [],  # False Positives
                'FN': [],  # False Negatives
                'TN': [] if self.track_true_negatives else None  # True Negatives
            } for i in range(self.num_classes)
        }
        self.error_count_per_class = {
            i: {
                'TP': 0,  # True Positives
                'FP': 0,  # False Positives
                'FN': 0,  # False Negatives
                'TN': 0 if self.track_true_negatives else None  # True Negatives
            } for i in range(self.num_classes)
        }