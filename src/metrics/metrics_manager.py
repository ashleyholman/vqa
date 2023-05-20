import boto3
import time
from botocore.exceptions import ClientError
from datetime import datetime
from decimal import Decimal

from src.metrics.performance_tracker import PerformanceMetrics

class MetricsManager:
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table('vqa')

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

    def store_performance_metrics(self, model_name, dataset_type, epoch, metrics: PerformanceMetrics, overwriteExisting=True):
        pk = f"{metrics.source}:{model_name}:{dataset_type}"
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
        self.table.put_item(
           Item={
                'PK': pk,
                'SK': sk,
                'model_name': model_name,
                'dataset_type': dataset_type,
                'epoch': epoch,
                'accuracy': Decimal(f"{metrics.accuracy:.10f}"),
                'top_5_acc': Decimal(f"{metrics.top_5_accuracy:.10f}"),
                'loss': Decimal(f"{metrics.loss:.10f}"),
                'timestamp': timestamp
            }
        )

    def get_performance_metrics(self, model_name, dataset_type):
        pk = f"test_result:{model_name}:{dataset_type}"
        return self._get_metrics(pk)

    def get_training_metrics(self, model_name, dataset_type):
        pk = f"training_loss:{model_name}:{dataset_type}"
        return self._get_metrics(pk)

    def _get_metrics(self, pk):
        metrics = []
        response = self.table.query(KeyConditionExpression='PK = :pk', ExpressionAttributeValues={':pk': pk})
        for item in response.get('Items', []):
            metrics.append({key: value for key, value in item.items() if key not in ['PK', 'SK']})
        return metrics