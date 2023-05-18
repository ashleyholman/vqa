import os
import json
import torch
import boto3
import shutil
from botocore.exceptions import NoCredentialsError

from torch.optim import Adam

from src.data.vqa_dataset import VQADataset
from src.models.vqa_model import VQAModel
from src.snapshots.snapshot import Snapshot

class SnapshotNotFoundException(Exception):
    pass

class InvalidSnapshotException(Exception):
    pass

class VQASnapshotManager:
    LOCAL_CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../snapshots')
    S3_BUCKET = 'vqa'

    def __init__(self):
        self.s3_client = boto3.client('s3')

        # ensure caching dir exists
        os.makedirs(self.LOCAL_CACHE_DIR, exist_ok=True)

    def load_snapshot(self, snapshot_name, dataset_type):
        self._populate_cache(snapshot_name)

        # Load the metadata
        with open(os.path.join(self.LOCAL_CACHE_DIR, snapshot_name, "metadata.json"), 'r') as f:
            metadata = json.load(f)

        # Ensure the dataset type matches
        if metadata['settype'] != dataset_type:
            raise ValueError(f"Snapshot '{snapshot_name}' was trained on dataset type '{metadata['settype']}', but dataset type '{dataset_type}' was requested.")

        # Load dataset
        dataset = VQADataset(settype=dataset_type, answer_classes=metadata['answer_classes'])

        # Initialize the model
        model = VQAModel(len(dataset.answer_classes))

        # Load model weights
        state_dict = torch.load(os.path.join(self.LOCAL_CACHE_DIR, snapshot_name, "model_weights.pth"))

        model.load_state_dict(state_dict, strict=not metadata['lightweight'])

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
        try:
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
                'loss': loss.item()
            }

            with open(os.path.join(self.LOCAL_CACHE_DIR, snapshot_name, "metadata.json"), 'w') as f:
                json.dump(metadata, f)

            if not skipS3Storage:
                self._save_to_s3(snapshot_name)
        except Exception as e:
            print(f'Failed to save snapshot: {e}')

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