import os
import json
import torch
import boto3
import shutil

from botocore.exceptions import NoCredentialsError
from datetime import datetime
from decimal import Decimal
from torch.optim import Adam

from src.data.vqa_dataset import VQADataset
from src.models.vqa_model import VQAModel
from src.snapshots.snapshot import Snapshot
from src.util.dynamodb_helper import DynamoDBHelper

class SnapshotNotFoundException(Exception):
    pass

class InvalidSnapshotException(Exception):
    pass

class VQASnapshotManager:
    LOCAL_CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../snapshots')
    S3_BUCKET = 'vqa-ap-southeast-1'

    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.ddb_helper = DynamoDBHelper()

        # ensure caching dir exists
        os.makedirs(self.LOCAL_CACHE_DIR, exist_ok=True)

    def load_snapshot(self, snapshot_name, dataset_type, device):
        self._populate_cache(snapshot_name)

        # Load the metadata
        with open(os.path.join(self.LOCAL_CACHE_DIR, snapshot_name, "metadata.json"), 'r') as f:
            metadata = json.load(f)

        # Load dataset
        dataset = VQADataset(settype=dataset_type, answer_classes=metadata['answer_classes'])

        # Initialize the model
        model = VQAModel(len(dataset.answer_classes))

        # Load model weights
        state_dict = torch.load(os.path.join(self.LOCAL_CACHE_DIR, snapshot_name, "model_weights.pth"), map_location=device)

        model.load_state_dict(state_dict, strict=not metadata['lightweight'])

        # Move the model to the requested device.
        # We need to do this before initializing the optimizer so that the
        # optimizer's tensors end up on the correct device.
        model.to(device)

        # Load the optimizer's state dict
        optimizer_state_dict = None
        if os.path.exists(os.path.join(self.LOCAL_CACHE_DIR, snapshot_name, "optimizer_state.pth")):
            optimizer_state_dict = torch.load(os.path.join(self.LOCAL_CACHE_DIR, snapshot_name, "optimizer_state.pth"))
        else:
            raise InvalidSnapshotException(f"Snapshot '{snapshot_name}' does not contain an optimizer state dict.")

        # Re-create the optimizer and load the state dict
        optimizer = Adam(model.parameters())
        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)

        return Snapshot(model, dataset, optimizer, metadata)

    def save_snapshot(self, snapshot_name, model, optimizer, dataset, epoch, loss, lightweight=False, skipS3Storage=False):
        # ensure snapshot dir exists
        os.makedirs(os.path.join(self.LOCAL_CACHE_DIR, snapshot_name), exist_ok=True)

        state_dict = model.state_dict()

        if lightweight:
            keys_to_remove = [key for key in state_dict if not key.startswith(('vit_transform', 'bert_transform', 'head'))]
            for key in keys_to_remove:
                state_dict.pop(key)

        torch.save(state_dict, os.path.join(self.LOCAL_CACHE_DIR, snapshot_name, "model_weights.pth"))

        # Save optimizer state
        torch.save(optimizer.state_dict(), os.path.join(self.LOCAL_CACHE_DIR, snapshot_name, "optimizer_state.pth"))

        metadata = {
            'settype': dataset.settype,
            'answer_classes': dataset.answer_classes,
            'lightweight': lightweight,
            'model_version': model.MODEL_NAME,
            'epoch': epoch,
            'loss': loss,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        with open(os.path.join(self.LOCAL_CACHE_DIR, snapshot_name, "metadata.json"), 'w') as f:
            json.dump(metadata, f)

        if skipS3Storage:
            # Skip storing in S3, and also skip storing the associated DDB record
            return

        # Save to S3
        self._save_to_s3(snapshot_name)

        # Before saving to DDB, convert any float values in the metadata to Decimal
        for key, value in metadata.items():
            if isinstance(value, float):
                metadata[key] = Decimal(str(value))

        # Remove answer class and add snapshot name
        metadata.pop('answer_classes')
        metadata['snapshot_name'] = snapshot_name

        # Insert DDB record
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pk = f"snapshot:{model.MODEL_NAME}:{dataset.settype}"
        sk = f"{epoch}:{timestamp}"  # Using epoch number
        self.ddb_helper.put_item(pk, sk, metadata)

    def list_snapshots(self):
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.S3_BUCKET,
                Prefix='snapshots/'
            )

            snapshot_names = set()
            for obj in response.get('Contents', []):
                prefix, snapshot_name, _ = obj['Key'].split('/', 2)
                if prefix == 'snapshots':
                    snapshot_names.add(snapshot_name)

            return list(snapshot_names)
        except Exception as e:
            print(f"Failed to list snapshots: {e}")
            return []

    def delete_snapshot(self, snapshot_name):
        # Safety check to ensure snapshot_name is not empty
        if not snapshot_name:
            raise ValueError("Snapshot name is required")

        local_snapshot_dir = os.path.join(self.LOCAL_CACHE_DIR, snapshot_name)

        # Remove from local cache
        if os.path.exists(local_snapshot_dir):
            shutil.rmtree(local_snapshot_dir)

        # Remove from S3
        s3_bucket = self.S3_BUCKET
        s3_prefix = f"snapshots/{snapshot_name}/"

        s3_resource = boto3.resource('s3')

        s3_bucket_resource = s3_resource.Bucket(s3_bucket)

        # Iterate over all objects with the given prefix and delete
        for obj in s3_bucket_resource.objects.filter(Prefix=s3_prefix):
            s3_resource.Object(s3_bucket_resource.name, obj.key).delete()

    def _populate_cache(self, snapshot_name):
        local_snapshot_path = os.path.join(self.LOCAL_CACHE_DIR, snapshot_name)
        if not (os.path.exists(os.path.join(local_snapshot_path, "model_weights.pth")) and 
                os.path.exists(os.path.join(local_snapshot_path, "metadata.json")) and
                os.path.exists(os.path.join(local_snapshot_path, "optimizer_state.pth"))):
            self._load_from_s3(snapshot_name)

    def _load_from_s3(self, snapshot_name):
        local_snapshot_dir = os.path.join(self.LOCAL_CACHE_DIR, snapshot_name)

        # Make a new local directory for the snapshot files
        os.makedirs(local_snapshot_dir, exist_ok=True)

        # Attempt to download the snapshot
        try:
            for filename in ["model_weights.pth", "optimizer_state.pth", "metadata.json"]:
                key = f"snapshots/{snapshot_name}/{filename}"
                self.s3_client.download_file(self.S3_BUCKET, key, os.path.join(local_snapshot_dir, filename))
        except Exception as e:
            print(f"Failed to download snapshot {snapshot_name}: {e}")

            # If download fails, delete the snapshot directory
            shutil.rmtree(local_snapshot_dir)
            raise SnapshotNotFoundException(f"Snapshot '{snapshot_name}' not found.")

    def _save_to_s3(self, snapshot_name):
        try:
            for filename in ["model_weights.pth", "optimizer_state.pth", "metadata.json"]:
                key = f"snapshots/{snapshot_name}/{filename}"
                self.s3_client.upload_file(os.path.join(self.LOCAL_CACHE_DIR, snapshot_name, filename), self.S3_BUCKET, key)
        except NoCredentialsError:
            print("Credentials not available")