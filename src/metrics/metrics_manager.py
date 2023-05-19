import boto3
import time
from botocore.exceptions import ClientError
from datetime import datetime
from decimal import Decimal

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

    def store_performance_metrics(self, model_name, dataset_type, epoch, accuracy, top_5_acc):
        pk = f"test_result:{model_name}:{dataset_type}"
        sk = str(epoch)

        # Fetch existing metrics
        existing_metrics = self._get_ddb_item(pk, sk)

        # If there are existing metrics, only update if current run performance is better
        if existing_metrics is not None:
            if existing_metrics['accuracy'] >= accuracy:
                print(f"An existing performance record exists with equal or better "
                      f"performance (accuracy={existing_metrics['accuracy']}%) from "
                      f"{existing_metrics['timestamp']}. Not updating.  If you want to force "
                      f"an update, delete the existing record first.")
                return
            else:
                print(f"Found existing performance record with worse performance "
                      f"(accuracy={existing_metrics['accuracy']}%) from {existing_metrics['timestamp']}. "
                      f"Overwriting with new performance metrics.")

        # Save metrics into DynamoDB
        timestamp = datetime.utcnow().isoformat()
        self.table.put_item(
           Item={
                'PK': pk,
                'SK': sk,
                'model_name': model_name,
                'dataset_type': dataset_type,
                'epoch': epoch,
                'accuracy': Decimal(f"{accuracy:.10f}"),
                'top_5_acc': Decimal(f"{top_5_acc:.10f}"),
                'timestamp': timestamp
            }
        )
    def store_training_metrics(self, model_name, dataset_type, epoch, loss):
        pk = f"training_loss:{model_name}:{dataset_type}"
        sk = str(epoch)

        # Save metrics into DynamoDB.  Any existing record for the
        # same model/dataset/epoch will be overwritten.
        timestamp = datetime.utcnow().isoformat()
        self.table.put_item(
           Item={
                'PK': pk,
                'SK': sk,
                'model_name': model_name,
                'dataset_type': dataset_type,
                'epoch': epoch,
                'loss': Decimal(f"{loss:.10f}"),
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