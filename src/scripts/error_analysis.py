import argparse
import torch
import json
from src.data.vqa_dataset import VQADataset
from src.models.model_configuration import ModelConfiguration
from src.models.vqa_model import VQAModel
from src.snapshots.vqa_snapshot_manager import VQASnapshotManager
from src.util.model_tester import ModelTester
from src.util.run_manager import RunManager
from src.metrics.error_tracker import ErrorTracker

class ErrorAnalysisRun:
    def __init__(self, args):
        self.run_manager = RunManager()
        self.snapshot_manager = VQASnapshotManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_id = args.run_id
        self.run = self.run_manager.get_run(self.run_id)
        self.config = ModelConfiguration.from_json_string(self.run.get("config"))
        self.num_dataloader_workers = args.num_dataloader_workers
        self.no_progress_bar = args.no_progress_bar

        if args.use_mini_dataset:
            self.validation_dataset_type = 'mini'
        else:
            self.validation_dataset_type = 'validation'

        print(f"Run ID: {self.run_id}")
        print(f"Validation dataset: {self.validation_dataset_type}")

    def load_snapshot(self):
        snapshot_name = self.run.get("snapshot_name")
        print(f"Run record: {self.run}")
        print(f"Loading snapshot '{snapshot_name}'")
        snapshot = self.snapshot_manager.load_snapshot(snapshot_name, self.validation_dataset_type, self.device)
        self.model = snapshot.get_model()
        self.validation_dataset = snapshot.get_dataset()
        self.model.to(self.device)
        print(f"Model architecture: ")
        print(self.model)

    def perform_validation_pass(self):
        error_analysis_tracker = ErrorTracker(
            num_classes=len(self.validation_dataset.answer_classes),
            max_samples_per_class=50,
            track_true_negatives=False,
            topk = 5,
            topk_as_positive=False)
        model_tester = ModelTester(self.config, self.validation_dataset, self.num_dataloader_workers)
        model_tester.test(self.model, None, error_analysis_tracker, self.device, self.no_progress_bar)

        return error_analysis_tracker.get_instance_data()

    def save_to_file(self, data, filename):
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Data saved to {filename}")

    def execute(self):
        self.load_snapshot()
        instance_data, error_count = self.perform_validation_pass()
        self.save_to_file({"instance_data": instance_data, "error_count": error_count}, "error_analysis_data.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Error Analysis Tracker')
    parser.add_argument('--run-id', required=True, help='ID of the run to load')
    parser.add_argument('--use-mini-dataset', default=False, action='store_true', help='Use mini dataset for validation')
    parser.add_argument('--num-dataloader-workers', type=int, default=1, help='Number of dataloader workers')
    parser.add_argument('--no-progress-bar', action='store_true', help='Use this flag to disable the progress bar during inference')
    args = parser.parse_args()

    error_analysis_run = ErrorAnalysisRun(args)
    error_analysis_run.execute()
