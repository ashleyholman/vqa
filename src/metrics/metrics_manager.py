from datetime import datetime
from decimal import Decimal

from src.util.dynamodb_helper import DynamoDBHelper

class MetricsManager:
    def __init__(self, source: str):
        self.ddb_helper = DynamoDBHelper()
        self.source = source

    def store_performance_metrics(self, model_name, dataset_type, epoch, metrics: dict, overwriteExisting=True):
        pk = f"{self.source}:{model_name}:{dataset_type}"
        sk = str(epoch)

        if not overwriteExisting:
            # Check for existing metrics record for this PK/SK
            existing_metrics = self.ddb_helper.get_item(pk, sk)
            if existing_metrics:
                print(f"WARNING: Found existing metrics record from {existing_metrics['timestamp']}."
                      f" New record will not be stored. Delete the existing record if you want to"
                      f" store a new one.")
                return

        # Save metrics into DynamoDB
        timestamp = datetime.utcnow().isoformat()

        metrics_item = {
            'model_name': model_name,
            'dataset_type': dataset_type,
            'epoch': epoch,
            'timestamp': timestamp
        }

        for metric_name, metric_value in metrics.items():
            if metric_name in metrics_item:
                print(f"WARNING: Metric name '{metric_name}' is a reserved key and cannot be overwritten. Skipping this metric.")
                continue
            metrics_item[metric_name] = Decimal(f"{metric_value:.10f}")

        self.ddb_helper.put_item(pk, sk, metrics_item)

    def get_metrics(self, source, model_name, dataset_type):
        pk = f"{source}:{model_name}:{dataset_type}"
        return self._get_metrics(pk)

    def _get_metrics(self, pk):
        metrics = []
        items = self.ddb_helper.query(pk)
        for item in items:
            metrics.append({key: value for key, value in item.items() if key not in ['PK', 'SK']})
        return metrics