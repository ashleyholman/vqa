import argparse

from src.models.model_configuration import ModelConfiguration
from src.util.model_trainer import ModelTrainer

def train_model(args):
    config = ModelConfiguration()
    model_trainer = ModelTrainer(config, args.num_dataloader_workers, args.dataset_type, args.from_snapshot)
    model_trainer.train(args.num_epochs, args.skip_s3_storage, args.lightweight_snapshots, args.no_progress_bar)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VQA model')
    parser.add_argument('--num-dataloader-workers', type=int, default=1, help='Number of dataloader workers')
    parser.add_argument('--dataset-type', type=str, default='train', help='Dataset type to train on (train, validation, mini)')
    parser.add_argument('--num-epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--from-snapshot', type=str, help="Snapshot name to load the model and dataset from.")
    parser.add_argument('--lightweight-snapshots', action='store_true', help="Use this flag to save lightweight snapshots only (doesn't save pretrained bert or vit weights)")
    parser.add_argument('--skip-s3-storage', action='store_true', help='Use this flag to skip storing new snapshots in S3')
    parser.add_argument('--no-progress-bar', action='store_true', help='Use this flag to disable the progress bar during training')
    args = parser.parse_args()
    train_model(args)