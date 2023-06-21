import csv
import decimal
import json
import os
import tempfile
import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib.ticker import MaxNLocator
from urllib.parse import urljoin
from urllib.request import pathname2url
from src.data.vqa_dataset import VQADataset

from src.metrics.metrics_manager import MetricsManager
from src.models.model_configuration import ModelConfiguration
from src.util.s3_helper import S3Helper

class GraphGenerator():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datasets = {}

    def __init__(self):
        self.metrics_manager = MetricsManager('GraphGenerator')
        self.s3_helper = S3Helper()

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

    def get_dataset(self, dataset_type):
        if dataset_type not in self.datasets:
            if dataset_type == 'validation':
                questions_json_file = VQADataset.VALIDATION_QUESTIONS_JSON_FILE_NAME
            elif dataset_type == 'mini':
                questions_json_file = VQADataset.MINI_QUESTIONS_JSON_FILE_NAME

            self.datasets[dataset_type] = self.load_dataset(questions_json_file)

        return self.datasets[dataset_type]

    def load_dataset(self, questions_json_file):
        dataset = { 'questions' : {} }

        with open(questions_json_file, 'r') as f:
            jsondata = json.load(f)
            for question in jsondata['questions']:
                dataset['questions'][question['question_id']] = {
                    'question_text': question['question'],
                    'image_id': question['image_id']
                }

        return dataset

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

            run_web_dir = f"{self.PROJECT_ROOT}/web-frontend/public/data/run/{run_id}"

            # determine the maximum epoch number that has finished training and validation.
            max_trained_epoch = None
            for epoch, metric_data in sorted(metrics.items(), reverse=True):
                # check if this epoch has both training and validation metrics
                # otherwise, it might not have finished validation yet, in which
                # case we can't use it to set the final_* values.
                if any(key.startswith(metrics_prefix) for key in metric_data.keys()):
                    max_trained_epoch = epoch
                    break

            if max_trained_epoch is None:
                print(f"No completed epoch found for run: {run_id}.  Skipping.")
                continue

            print(f"Max trained epoch: {max_trained_epoch}")

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

            has_error_analysis = False
            # check for existance of error analysis data in run record
            if 'error_analysis_s3_url' in run and 'answer_classes' in run:
                output_dir = f"{run_web_dir}/error_analysis"
                if not os.path.exists(output_dir) or regenerate_existing or run['run_status'] == 'IN_PROGRESS':
                    # Create output directory if it doesn't exist
                    os.makedirs(output_dir, exist_ok=True)

                    print(f"Fetching error analysis data from: {run['error_analysis_s3_url']}")
                    # Fetch metadata directly from S3 to a temporary file
                    with tempfile.NamedTemporaryFile(suffix="_error_analysis.json") as temp_json_file:
                        self.s3_helper.download_file_from_url(run['error_analysis_s3_url'], temp_json_file.name)
                        temp_json_file.flush()
                        # Read the error analysis data
                        with open(temp_json_file.name, 'r') as f:
                            error_analysis_data = json.load(f)
                            self.generate_error_analysis_json_for_web(run, error_analysis_data, output_dir)
                has_error_analysis = True

            index_entry['has_error_analysis'] = has_error_analysis
            index_metadata.append(index_entry)

            # now generate the main JSON file for this run (contains config and metrics)
            main_json_file = f"{run_web_dir}/main.json"
            if not os.path.exists(main_json_file) or regenerate_existing or run['run_status'] == 'IN_PROGRESS':
                # Create output directory if it doesn't exist
                os.makedirs(run_web_dir, exist_ok=True)

                data = { 'run_id': run_id, 'config': json.loads(run.get('config', '{}')) }
                data['validation_dataset_type'] = run['validation_dataset_type']
                data['metrics'] = metrics
                data['has_error_analysis'] = has_error_analysis
                with open(main_json_file, 'w') as f:
                    json.dump(data, f, default=lambda obj: float(obj) if isinstance(obj, decimal.Decimal) else TypeError, indent=4)
                print(f"Generated main JSON file for run: {run_id} to outfile {main_json_file}")
            else:
                print(f"Main JSON file for run: {run_id} already exists, skipping...")
            json_files.append(main_json_file)

        # write out the runs.json file
        runs_file_name = f"{self.PROJECT_ROOT}/web-frontend/public/data/runs.json"
        with open(runs_file_name, 'w') as f:
            json.dump(index_metadata, f, default=lambda obj: float(obj) if isinstance(obj, decimal.Decimal) else TypeError, indent=4)
            json_files.append(runs_file_name)

        return json_files

    def generate_error_analysis_json_for_web(self, run, error_analysis_data, json_outdir):
        ''' Generate a separate JSON file for each class, containing sets of
        example cases for true positives, false positives, and false negatives.
        '''
        # if it's a list, this is legacy format that never made it to prod, skip it
        run_id = run['run_id']

        if isinstance(error_analysis_data, list):
            print(f"Skipping legacy format error analysis data for run: {run_id}")
            return

        dataset = self.get_dataset(run['validation_dataset_type'])

        # sample questions contain 'question_id' only.  We need to turn this
        # into question text, and an image ID so that the web app can display
        # the question to the user.
        for class_id, class_data in error_analysis_data["sample_questions"].items():
            json_for_web = {}
            for category, sample_questions in class_data.items():
                json_for_web[category] = { 'sample_questions': [] }
                if sample_questions is None:
                    # category not implemented (eg. true negatives)
                    continue

                for sample_question in sample_questions:
                    question_id = sample_question['question_id']
                    if question_id not in dataset['questions']:
                        print(f"Question ID {question_id} not found in dataset, ERROR.")
                        exit(1)
                    record = {
                        'question_text' : dataset['questions'][question_id]['question_text'],
                        'image_id' :  dataset['questions'][question_id]['image_id'],
                        'true_class' : sample_question['true_class'],
                        'predicted_classes' : sample_question['predicted_classes']
                    }
                    json_for_web[category]['sample_questions'].append(record)
            file_path = os.path.join(json_outdir, f"class_{class_id}.json")
            with open(file_path, mode='w', newline='') as file:
                json.dump(json_for_web, file)
            print(f"Saved error analysis data for class {class_id} to {file_path}")

        # now generate the summary.json file, containing answer classes and statistics
        summary_json_structure = {}

        for class_index, statistics in error_analysis_data['counts'].items():
            class_label = run['answer_classes'][int(class_index)]
            summary_json_structure[class_index] = {
                "class_label": class_label,
                "statistics": statistics
            }

        summary_json_file = os.path.join(json_outdir, "summary.json")
        with open(summary_json_file, mode='w') as file:
            json.dump(summary_json_structure, file, indent=4)
            print(f"Saved error analysis summary data to {summary_json_file}")