import argparse
import os
from src.data.vqa_dataset import VQADataset

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/')

def main():
    parser = argparse.ArgumentParser(description='Generate and save embeddings for a dataset.')
    parser.add_argument('--dataset-type', type=str, default='train',
                        help='Dataset type to generate the embeddings for (train, validation, mini, etc).')
    parser.add_argument('--num-dataloader-workers', type=int, default=1, help='Number of dataloader workers')

    args = parser.parse_args()

    dataset_type = args.dataset_type
    save_path = os.path.join(DATA_DIR, f'{dataset_type}_embeddings.pt')

    print(f'Loading {dataset_type} dataset...')
    dataset = VQADataset(dataset_type)
    print(f'Dataset loaded.')

    print(f'Generating and saving embeddings...')
    dataset.generate_and_save_all_embeddings(args.num_dataloader_workers)
    print('Done.')

if __name__ == '__main__':
    main()