import argparse
import datetime
import hashlib
import os
import time
import uuid
import torch
import subprocess

from torch.optim import Adam
from datetime import datetime


from src.data.vqa_dataset import VQADataset
from src.metrics.metrics_manager import MetricsManager
from src.metrics.performance_tracker import PerformanceTracker

from src.models.model_configuration import ModelConfiguration
from src.models.vqa_model import VQAModel
from src.snapshots.vqa_snapshot_manager import VQASnapshotManager
from src.util.dynamodb_helper import DynamoDBHelper
from src.util.model_tester import ModelTester
from src.util.model_trainer import ModelTrainer

TRAINING_METRICS_SOURCE = "train_model"
VALIDATION_METRICS_SOURCE = "test_model"

class Run:
    def __init__(self, config, args):
        self.config = config
        self.snapshot_manager = VQASnapshotManager()
        self.ddb_helper = DynamoDBHelper()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.skip_s3_storage = args.skip_s3_storage
        self.no_progress_bar = args.no_progress_bar

        # allow --max-epochs to override the config setting
        self.config.max_epochs = args.max_epochs

        if args.use_mini_dataset:
            self.training_dataset_type = 'mini'
            self.validation_dataset_type = 'mini'
        else:
            self.training_dataset_type = 'train'
            self.validation_dataset_type = 'validation'

        # Build a "state hash" which is a unique hash representing the state of the configuration set and code.
        self.state_hash = self._get_state_hash()

        print(f"State hash: {self.state_hash}")

        # call _get_or_create_run() to set self.run, self.start_epoch, and self.snapshot_name
        self._get_or_create_run()

        print("===== Ready to start Run =====")
        print(f"Run ID: {self.run_id}")
        print(f"Start epoch: {self.start_epoch}")
        print(f"Snapshot name: {self.snapshot_name}")
        print(f"Max epochs: {self.config.max_epochs}")
        print("==============================")

    def set_restore_point(self, trained_until_epoch):
        # Save a snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"snapshot_{self.config.model_name}_{self.training_dataset_type}_epoch_{trained_until_epoch}_{timestamp}"
        print(f"Saving snapshot '{snapshot_name}'")

        # Save the model state, optimizer state, and answer classes
        self.snapshot_manager.save_snapshot(snapshot_name, self.model, self.optimizer, self.training_dataset, trained_until_epoch, lightweight=False, skipS3Storage=self.skip_s3_storage)

        # Update our unfinished-run checkpoint in DDB
        column_values = {
            'trained_until_epoch': trained_until_epoch,
            'snapshot_name': snapshot_name,
            'updated_at': datetime.now().isoformat()
        }
        self.ddb_helper.update_item(f"unfinished-run:{self.state_hash}", self.run_id, column_values)

    def _get_state_hash(self):
        '''Get a hash of the current state of the code and config.

        Typically the config values come from the model-config.yaml file, which is
        committed in git, so sometimes the addition of the config hash may be
        redundant here.  However, the ModelConfiguration class supports overwriting
        of configuration settings in-memory, for example like we do with
        config.max_epochs below, overriding it with the --max-epochs CLI argument.
        '''

        # build hash of configuration state
        print(f"Config json: {self.config.to_json_string()}")
        config_hash = hashlib.sha256(self.config.to_json_string().encode()).hexdigest()

        # obtain git commit hash of HEAD
        repo_root = os.path.join(os.path.dirname(__file__), '../../')
        git_commit_hash = subprocess.check_output(['git', '-C', repo_root, 'rev-parse', 'HEAD']).strip().decode()

        combined_hash_input = f"{config_hash}:{git_commit_hash}:{self.training_dataset_type}:{self.validation_dataset_type}"
        print(f"Combined hash input: {combined_hash_input}")

        # Return the first 20 characters of the combined hash.  This is sufficiently unique for our use case.
        return hashlib.sha256(combined_hash_input.encode()).hexdigest()[:20]

    def _get_or_create_run(self):
        '''Check DDB for an unfinished run for this state hash

        Returns a tuple of (run_id, starting_epoch, snapshot_name) if there is an unfinished run,
        '''
        pk = f"unfinished-run:{self.state_hash}"
        unfinished_runs = self.ddb_helper.query(pk)

        # if there are unfinished runs
        print("Unfinished runs: ", unfinished_runs)
        if len(unfinished_runs) > 0:
            print(f"Found {len(unfinished_runs)} unfinished runs for state hash '{self.state_hash}'")
            print(unfinished_runs)

            # return the first one
            self.run_id = unfinished_runs[0]['run_id']
            self.start_epoch = unfinished_runs[0]['trained_until_epoch']+1
            self.snapshot_name = unfinished_runs[0]['snapshot_name']
        else:
            # generate a unique UUID for a new run
            self.run_id = str(uuid.uuid4())
            self.start_epoch = 1
            self.snapshot_name = None

            # insert an "unfinished-run" DDB record for this new run
            column_values = {
                'run_id': self.run_id,
                'training_dataset_type': self.training_dataset_type,
                'validation_dataset_type': self.validation_dataset_type,
                'max_epochs': self.config.max_epochs,
                'state_hash': self.state_hash,
                'trained_until_epoch': 0,
                'snapshot_name': None,
                'config': self.config.to_json_string(),
                'started_at': datetime.now().isoformat()
            }
            print("INSERTING DDB")
            self.ddb_helper.put_item(pk, self.run_id, column_values)

    def run(self):
        '''Run the training and validation process.'''
        if self.snapshot_name:
            print(f"Resuming run '{self.run_id}' from epoch {self.start_epoch} with snapshot '{self.snapshot_name}'")
            snapshot = self.snapshot_manager.load_snapshot(self.snapshot_name, self.training_dataset_type, self.device)
        else:
            print(f"Starting new run '{self.run_id}' from epoch {self.start_epoch}")
            snapshot = None

        num_dataloader_workers = args.num_dataloader_workers

        training_metrics_manager = MetricsManager(TRAINING_METRICS_SOURCE)
        validation_metrics_manager = MetricsManager(VALIDATION_METRICS_SOURCE)

        if snapshot:
            # Since we're resuming training, the dataset and model version we're training
            # on should be the same as the ones in the snapshot, otherwise we'll get
            # unexpected results.
            if (snapshot.get_metadata()["settype"] != self.training_dataset_type):
                print(f"ERROR: Dataset type '{self.training_dataset_type}' does not match the dataset type in the snapshot '{snapshot.get_metadata()['dataset_type']}'.")
                return

            if (snapshot.get_metadata()["model_version"] != self.config.model_name):
                print(f"ERROR: Model name '{self.config.model_name}' does not match the model name in the snapshot '{snapshot.get_metadata()['model_name']}'.")
                return

            print(f"Using snapshot: {snapshot.get_name()}")
            self.training_dataset = snapshot.get_dataset()
            self.model = snapshot.get_model()
            self.optimizer = snapshot.get_optimizer()

            # The epoch number stored in the snapshot represents
            # the last completed training epoch, so resume training from epoch+1
            start_epoch = snapshot.get_metadata()["epoch"]+1
            if (start_epoch > self.config.max_epochs):
                print(f"Snapshot '{snapshot.get_name()}' already trained for {start_epoch-1} epochs.  Nothing to do.")
                return

            if (snapshot.isLightweight()):
                print(f"WARNING: Snapshot '{snapshot.get_name()}' is lightweight, meaning the pretrained "
                      "weights of the BERT and ViT models are not included in the snapshot.  Those models "
                      "will be initialized from huggingface.  If the underlying weights of these models "
                      "have changed since the snapshot was created, the model may not train correctly.")

            print(f"Resuming training from epoch {start_epoch}.")
        else:
            # load training dataset
            print("Loading training dataset...")
            self.training_dataset = VQADataset(self.training_dataset_type)

            # load model
            print("Answer classes: ", self.training_dataset.answer_classes)
            self.model = VQAModel(self.training_dataset.answer_classes)

            # Create a new optimizer
            self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)

            # epoch's are 1-indexed for ease of understanding by the user
            start_epoch = 1

        self.model.to(self.device)
        if self.config.use_answer_embeddings:
            # FIXME: Handle this in snapshot manager or VQADataset class
            self.model.answer_embeddings = self.model.answer_embeddings.to(self.device)

        # Load the validation dataset, using the same answer classes as the training set.
        print("Loading validation dataset...")
        self.validation_dataset = VQADataset(self.validation_dataset_type, self.training_dataset.answer_classes)

        model_trainer = ModelTrainer(self.config, self.model, self.training_dataset, self.optimizer, num_dataloader_workers)
        model_tester = ModelTester(self.config, self.validation_dataset, num_dataloader_workers)
        performance_tracker_reusable = PerformanceTracker()

        for epoch in range(start_epoch, self.config.max_epochs+1):
            print(f"Epoch {epoch}/{self.config.max_epochs}")

            is_snapshot_epoch = (epoch % self.config.snapshot_every_epochs == 0)
            is_metrics_epoch = (epoch % self.config.metrics_every_epochs == 0)
            performance_tracker = None

            start_time = time.time()

            if is_metrics_epoch:
                performance_tracker = performance_tracker_reusable
                performance_tracker.reset()

            print("Begin training...")
            model_trainer.train_one_epoch(performance_tracker)

            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch} completed training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, took {elapsed_time/60:.2f} minutes.")

            if is_metrics_epoch:
                performance_tracker.print_report()
                training_metrics_manager.store_performance_metrics(self.config.model_name, self.training_dataset_type, epoch, performance_tracker.get_metrics(), True, self.run_id)

            if is_snapshot_epoch:
                # Update our unfinished-run checkpoint in DDB
                self.set_restore_point(epoch)

            if not is_metrics_epoch:
                # if it's not a metrics epoch, we don't need to do validation.  Continue to the next epoch.
                continue

            print("Validating model...")
            performance_tracker.reset()
            model_tester.test(self.model, performance_tracker, self.device, self.no_progress_bar)

            # Print performance report
            performance_tracker.print_report()

            # store metrics
            validation_metrics_manager.store_performance_metrics(self.config.model_name, self.validation_dataset_type, epoch, performance_tracker.get_metrics(), True, self.run_id)

        print("Run complete.")

if __name__ == "__main__":
    config = ModelConfiguration()
    parser = argparse.ArgumentParser(description='Execute a training and validation run.')
    parser.add_argument('--num-dataloader-workers', type=int, default=1, help='Number of dataloader workers')
    parser.add_argument('--max-epochs', type=int, default=config.max_epochs, help='The maximum number of epochs to train for.')
    parser.add_argument('--skip-s3-storage', action='store_true', help='Use this flag to skip storing new snapshots in S3')
    parser.add_argument('--no-progress-bar', action='store_true', help='Use this flag to disable the progress bar during training')
    parser.add_argument('--use-mini-dataset', action='store_true', help='For local testing.  Both training and validation are done on the mini dataset.')

    args = parser.parse_args()
    run = Run(config, args)
    run.run()