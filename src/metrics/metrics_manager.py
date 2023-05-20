import boto3
import time
from botocore.exceptions import ClientError
from datetime import datetime
from decimal import Decimal

class MetricsManager:
    def __init__(self, source: str):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table('vqa')
        self.source = source

    def _get_ddb_item(self, pk, sk):
        try:
            response = self.table.get_item(
                Key={
                    'PK': pk,
                    'SK': sk
                }
            )
        except ClientError as e:
            print(e.response['Error']['Message'])
        else:
            return response.get('Item')

    def store_performance_metrics(self, model_name, dataset_type, epoch, metrics: dict, overwriteExisting=True):
        pk = f"{self.source}:{model_name}:{dataset_type}"
        sk = str(epoch)

        if not overwriteExisting:
            # Check for existing metrics record for this PK/SK
            existing_metrics = self._get_ddb_item(pk, sk)
            if existing_metrics:
                print(f"WARNING: Found existing metrics record from {existing_metrics['timestamp']}."
                      f" New record will not be stored. Delete the existing record if you want to"
                      f" store a new one.")
                return

        # Save metrics into DynamoDB
        timestamp = datetime.utcnow().isoformat()

        metrics_item = {
            'PK': pk,
            'SK': sk,
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

        self.table.put_item(Item=metrics_item)

    def get_metrics(self, source, model_name, dataset_type):
        pk = f"{source}:{model_name}:{dataset_type}"
        return self._get_metrics(pk)

    def _get_metrics(self, pk):
        metrics = []
        response = self.table.query(KeyConditionExpression='PK = :pk', ExpressionAttributeValues={':pk': pk})
        for item in response.get('Items', []):
            metrics.append({key: value for key, value in item.items() if key not in ['PK', 'SK']})
        return metrics