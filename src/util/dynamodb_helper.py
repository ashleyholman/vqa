import boto3

from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

class DynamoDBHelper:
    TABLE_NAME = 'vqa'
    GSI_NAME = 'GSI'
    GSI_PK_COLUMN_NAME = 'GSI_PK'
    GSI_SK_COLUMN_NAME = 'GSI_SK'

    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(self.TABLE_NAME)

    def put_item(self, pk, sk, values: dict):
        try:
            item = {
                'PK': pk,
                'SK': sk,
                **values
            }
            self.table.put_item(Item=item)
        except ClientError as e:
            print(e.response['Error']['Message'])

    def get_item(self, pk, sk):
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

    def update_item(self, pk, sk, values: dict):
        try:
            expression = 'SET '
            expression_values = {}
            for key, value in values.items():
                expression += f"{key} = :{key}, "
                expression_values[f":{key}"] = value
            expression = expression[:-2]

            response = self.table.update_item(
                Key={
                    'PK': pk,
                    'SK': sk
                },
                UpdateExpression=expression,
                ExpressionAttributeValues=expression_values,
                ReturnValues='UPDATED_NEW'
            )
        except ClientError as e:
            print(e.response['Error']['Message'])
        else:
            return response.get('Attributes')

    def delete_item(self, pk, sk):
        try:
            self.table.delete_item(
                Key={
                    'PK': pk,
                    'SK': sk
                }
            )
        except ClientError as e:
            print(e.response['Error']['Message'])

    def query(self, pk):
        try:
            response = self.table.query(
                KeyConditionExpression='PK = :pk',
                ExpressionAttributeValues={':pk': pk}
            )
        except ClientError as e:
            print(e.response['Error']['Message'])
        else:
            return response.get('Items', [])
    
    def query_gsi(self, pk, limit=None, scan_index_forward=True):
        try:
            # Create the query parameters
            params = {
                'IndexName': self.GSI_NAME,
                'KeyConditionExpression': Key(self.GSI_PK_COLUMN_NAME).eq(pk),
                'ScanIndexForward': scan_index_forward
            }

            # Add the limit to the query parameters if one was provided
            if limit is not None:
                params['Limit'] = limit

            # Execute the query
            response = self.table.query(**params)

            # Return the list of items from the response
            return response['Items']

        except ClientError as e:
            print(e.response['Error']['Message'])