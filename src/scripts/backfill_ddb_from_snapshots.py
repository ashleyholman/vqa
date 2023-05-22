import argparse
import boto3
from decimal import Decimal
import json
import tempfile

from src.snapshots.vqa_snapshot_manager import VQASnapshotManager
from src.util.dynamodb_helper import DynamoDBHelper

S3_BUCKET = 'vqa-ap-southeast-1'

def backfill_ddb(dry_run=False):
    snapshot_manager = VQASnapshotManager()
    ddb_helper = DynamoDBHelper()
    s3_client = boto3.client('s3')

    # get the list of snapshots from S3
    snapshots = snapshot_manager.list_snapshots()

    for snapshot_name in snapshots:
        print(f"Fetching metadata for snapshot: {snapshot_name}")
        # Fetch metadata directly from S3 to a temporary file
        with tempfile.NamedTemporaryFile(suffix="_metadata.json") as temp_metadata_file:
            s3_client.download_file(
                S3_BUCKET, 
                f"snapshots/{snapshot_name}/metadata.json", 
                temp_metadata_file.name
            )

            # Read the metadata
            with open(temp_metadata_file.name, 'r') as f:
                metadata = json.load(f)

            # Convert any float values in the metadata to Decimal
            for key, value in metadata.items():
                if isinstance(value, float):
                    metadata[key] = Decimal(str(value))

            # Remove answer classes and add snapshot name to metadata
            metadata.pop('answer_classes')
            metadata['snapshot_name'] = snapshot_name

            if not metadata.get('timestamp'):
                metadata['timestamp'] = '_'.join(snapshot_name.split('_')[-2:])

            # Form primary key (pk) and sort key (sk) based on metadata
            pk = f"snapshot:{metadata['model_version']}:{metadata['settype']}"
            sk = f"{metadata['epoch']}:{metadata['timestamp']}" 
            print(f"PK: {pk}, SK: {sk}")

            # check if DDB record exists
            item = ddb_helper.get_item(pk, sk)
            if item is not None:
                print(f"Record exists in DynamoDB for snapshot: {snapshot_name}, skipping")
            else:
                if not dry_run:
                    # Insert DDB record
                    print(f"No record found in DDB for snapshot: {snapshot_name}, inserting: {metadata}")
                    ddb_helper.put_item(pk, sk, metadata)
                else:
                    print(f"Dry run mode - Would have inserted: {metadata} for snapshot: {snapshot_name}")

def main():
    parser = argparse.ArgumentParser(description='Backfill DynamoDB with existing VQA snapshots')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without actual DynamoDB insertions')

    args = parser.parse_args()

    backfill_ddb(dry_run=args.dry_run)

if __name__ == "__main__":
    main()