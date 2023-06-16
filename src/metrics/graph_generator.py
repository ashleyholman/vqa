import csv
import decimal
import json
import os
import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib.ticker import MaxNLocator
from urllib.parse import urljoin
from urllib.request import pathname2url

from src.metrics.metrics_manager import MetricsManager
from src.models.model_configuration import ModelConfiguration

class GraphGenerator():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def __init__(self):
        self.metrics_manager = MetricsManager('GraphGenerator')

    def __fetch_metrics(self, run_id, for_mini_dataset=False):
        if for_mini_dataset:
            datasets = {'mini': self.metrics_manager.get_metrics_by_run_id(run_id, 'mini')}
        else:
            datasets = {
                'training': self.metrics_manager.get_metrics_by_run_id(run_id, 'train'),
                'validation': self.metrics_manager.get_metrics_by_run_id(run_id, 'validation')
            }

        data = {}

        for prefix, metrics in datasets.items():
            for item in metrics:
                epoch = int(item['epoch'])
                if epoch not in data:
                    data[epoch] = {'epoch': epoch}
                for key, value in item.items():
                    if key not in ['epoch', 'dataset_type', 'timestamp', 'model_name']:
                        data[epoch][f'{prefix}_{key}'] = float(value)

        return data

    def generate_csv_for_run(self, run_id, for_mini_dataset=False):
        data = self.__fetch_metrics(run_id, for_mini_dataset)

        if not data:
            print(f"No metrics found for run: {run_id}")
            return

        output_dir_suffix = "_mini" if for_mini_dataset else ""
        output_name = f"run_{run_id}"
        output_dir = f"{self.PROJECT_ROOT}/graphs/{output_name}{output_dir_suffix}"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(output_dir, "results.csv")
        with open(file_path, mode='w', newline='') as file:
            header = ['epoch']

            # Collect all unique keys from the data
            for epoch in data.keys():
                for key in data[epoch].keys():
                    if key not in header:
                        header.append(key)

            writer = csv.writer(file)
            writer.writerow(header)

            for epoch in sorted(data.keys()):
                row = [data[epoch].get(key, '') for key in header]
                writer.writerow(row)

        return file_path, data

    def generate_json_files(self, runs, regenerate_existing=False):
        ''' Generate JSON files for each run in the list of runs provided.
            Also generate a runs.json file to be used by the index page that lists all runs.'''
        json_files = []
        index_metadata = []
        for run in runs:
            run_id = run['run_id']

            if run['validation_dataset_type'] == 'mini':
                is_mini_dataset = True
                metrics_prefix = 'mini_'
            else:
                is_mini_dataset = False
                metrics_prefix = 'validation_'

            metrics = self.__fetch_metrics(run_id, is_mini_dataset)

            # skip this run if no metrics were found
            if not metrics:
                print(f"No metrics found for run: {run_id}.  Skipping.")
                continue

            max_trained_epoch = max(metrics.keys())

            # copy specific keys from run to index_metadata
            index_entry = {}
            for key in ['run_id', 'started_at', 'run_status', 'config', 'validation_dataset_type']:
                index_entry[key] = run[key]
            index_entry['num_trained_epochs'] = max_trained_epoch

            # copy the last epoch's validation metrics to final_* entries
            for key, value in metrics[max_trained_epoch].items():
                if key.startswith(metrics_prefix):
                    new_key = 'final_' + key.split(metrics_prefix)[1]
                    index_entry[new_key] = value

            index_metadata.append(index_entry)

            # now generate the JSON file for this run
            json_file = f"{self.PROJECT_ROOT}/web-frontend/public/data/run_{run_id}.json"
            if not os.path.exists(json_file) or regenerate_existing:
                data = { 'run_id': run_id, 'config': json.loads(run.get('config', '{}')) }
                data['validation_dataset_type'] = run['validation_dataset_type']
                data['metrics'] = metrics
                with open(json_file, 'w') as f:
                    json.dump(data, f, default=lambda obj: float(obj) if isinstance(obj, decimal.Decimal) else TypeError, indent=4)
                print(f"Generated JSON file for run: {run_id} to outfile {json_file}")
            else:
                print(f"JSON file for run: {run_id} already exists, skipping...")
            json_files.append(json_file)

        # write out the runs.json file
        runs_file_name = f"{self.PROJECT_ROOT}/web-frontend/public/data/runs.json"
        with open(runs_file_name, 'w') as f:
            json.dump(index_metadata, f, default=lambda obj: float(obj) if isinstance(obj, decimal.Decimal) else TypeError, indent=4)
            json_files.append(runs_file_name)

        return json_files