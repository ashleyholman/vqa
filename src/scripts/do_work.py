import argparse
import os
import time
from src.snapshots.vqa_snapshot_manager import VQASnapshotManager
from src.metrics.metrics_manager import MetricsManager
from src.models.vqa_model import VQAModel

def get_work(model_name, dataset_type):
    print(f"Finding unvalidated snapshots for model '{model_name}' and dataset type '{dataset_type}'")

    snapshot_manager = VQASnapshotManager()
    metrics_manager = MetricsManager("do_work")

    # Get the list of snapshots for this model name that were trained on the given dataset type
    snapshots = snapshot_manager.list_snapshots(model_name, dataset_type)

    # Fetch all validation metrics for this model, ie. test results from the validation dataset
    validation_dataset = "validation"
    if dataset_type == 'mini':
        # for local testing only, the 'mini' dataset is both trained and validated on the same dataset.
        validation_dataset = 'mini'
    metrics = metrics_manager.get_metrics("test_model", model_name, validation_dataset)

    # Group snapshots by epoch, taking only the latest snapshot per epoch
    # At the same time, track the most recent snapshot and its epoch, so
    # we can return those too for resuming training.
    snapshots_by_epoch = {}
    latest_snapshot_epoch = -1
    latest_snapshot_name = None
    for snapshot_name, metadata in snapshots.items():
        epoch = metadata['epoch']
        if epoch not in snapshots_by_epoch or metadata['timestamp'] > snapshots_by_epoch[epoch]['timestamp']:
            snapshots_by_epoch[epoch] = metadata
            if epoch > latest_snapshot_epoch:
                latest_snapshot_epoch = epoch
                latest_snapshot_name = snapshot_name

    for metric in metrics:
        epoch = metric['epoch']
        if epoch in snapshots_by_epoch:
            snapshots_by_epoch[epoch]['validation_metrics'] = metric

    snapshots_to_validate = []

    # For each epoch's latest snapshot, print out the metrics
    for epoch, metadata in snapshots_by_epoch.items():
        snapshot_name = metadata['snapshot_name']
        if 'validation_metrics' not in metadata:
            snapshots_to_validate.append(snapshot_name)

    return snapshots_to_validate, latest_snapshot_epoch, latest_snapshot_name

# Note: When --dataset-type mini is passed, we'll both train and validate on the
# mini dataset. In production, we always train on the 'train' dataset and validate on
# the 'validation' dataset.
def validate_snapshot(snapshot_name, num_dataloader_workers, dataset_type):
    # Form the command to execute
    cmd = f"python ./src/scripts/test_model.py --from-snapshot {snapshot_name} --num-dataloader-workers {num_dataloader_workers} --no-progress-bar"
    if dataset_type == 'mini':
        cmd += " --dataset-type mini"

    print(f"Executing command: {cmd}")
    os.system(cmd)

def train_model(snapshot_name, target_epoch, num_dataloader_workers, dataset_type):
    # Form the base command to execute
    cmd = f"python ./src/scripts/train_model.py --num-dataloader-workers {num_dataloader_workers} --num-epochs {target_epoch} --no-progress-bar"

    # If snapshot_name is not None, add --from-snapshot to command
    if snapshot_name is not None:
        cmd += f" --from-snapshot {snapshot_name}"

    if dataset_type == 'mini':
        cmd += " --dataset-type mini --lightweight-snapshots"

    print(f"Executing command: {cmd}")
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser(description='Enter a continuous loop of training and validation of our current model version')
    parser.add_argument('--dataset-type', type=str, default='train', help='Dataset to train on.  Validation of snapshots will be done for unvalidated snapshots trained on this dataset.')
    parser.add_argument('--num-dataloader-workers', type=int, default=1, help='Number of dataloader workers to use for validation.')
    args = parser.parse_args()

    while True:
        main_loop(args)

        # Sleep 5 seconds between iterations
        time.sleep(5)

def main_loop(args):
    snapshots_to_validate, latest_snapshot_epoch, latest_snapshot_name = get_work(VQAModel.MODEL_NAME, args.dataset_type)
    for snapshot_name in snapshots_to_validate:
        print(f"Validating snapshot: {snapshot_name}")
        validate_snapshot(snapshot_name, args.num_dataloader_workers, args.dataset_type)

    print("No more snapshots to validate. Will resume training model for another 5 epochs.")
    train_model(latest_snapshot_name, latest_snapshot_epoch + 5, args.num_dataloader_workers, args.dataset_type)
    print("Done training.")

if __name__ == '__main__':
    main()