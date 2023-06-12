import csv
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

    def __plot_graphs(self, data, run_id, for_mini_dataset=False):
        plt.style.use('dark_background')
        epochs = sorted(data.keys())
        metrics = set([k.split('_', 1)[1] for k in data[epochs[0]].keys() if '_' in k])

        output_dir_suffix = "_mini" if for_mini_dataset else ""
        output_name = f"run_{run_id}"
        output_dir = f"{self.PROJECT_ROOT}/graphs/{output_name}{output_dir_suffix}"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        image_files = []
        metric_pairs = [('accuracy', 'top_5_accuracy'),
                        ('precision_macro', 'precision_micro'),
                        ('recall_macro', 'recall_micro'),
                        ('f1_score_macro', 'f1_score_micro')]

        # Determine the datasets to plot metrics for, and their corresponding colours
        dataset_prefixes = {'mini': 'pink'} if for_mini_dataset else {'training': 'yellow', 'validation': 'cyan'}

        # Plot 'loss' first as a larger graph
        if 'loss' in metrics:
            plt.figure(figsize=(10, 5))
            ax = plt.gca()
            for prefix, color in dataset_prefixes.items():
                ax.plot(epochs, [data[epoch].get(f'{prefix}_loss') for epoch in epochs], linestyle='-', marker='o', color=color, label=prefix)
            ax.set_xlabel('epochs')
            ax.set_ylabel('loss')
            ax.set_title('loss')
            ax.legend()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            image_file = f"loss.png"
            plt.savefig(os.path.join(output_dir, image_file))
            image_files.append(("loss", image_file))
            plt.close()

        # Plot the rest of the metrics
        for metric in metrics:
            if metric != 'loss':
                plt.figure(figsize=(5, 3))
                ax = plt.gca()
                for prefix, color in dataset_prefixes.items():
                    ax.plot(epochs, [data[epoch].get(f'{prefix}_{metric}') for epoch in epochs], linestyle='-', marker='o', color=color, label=prefix)
                ax.set_xlabel('Epochs')
                ax.set_ylabel(metric)
                ax.set_title(metric)
                ax.legend()
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.tight_layout()
                image_file = f"{metric}.png"
                plt.savefig(os.path.join(output_dir, image_file))
                image_files.append((metric, image_file))
                plt.close()

        with open(os.path.join(output_dir, "index.html"), "w") as f:
            f.write('<html>\n<head>\n<style>\nbody {background-color: #303030; display: flex; flex-wrap: wrap; justify-content: center;}\nimg {margin: 10px; max-width: 100%; height: auto;}\n.container {display: flex;}\n.container img {flex: 1;}\n</style>\n</head>\n<body>\n')

            # Add 'loss' graph at the top of the page, in a bigger size.
            loss_image = next((file for metric, file in image_files if metric == "loss"), None)
            if loss_image:
                f.write(f'<img src="{loss_image}" style="margin-left:auto; margin-right:auto;" />\n')

            # The rest of the metrics will be shown in a 2 column layout, pairing those which are related.
            for pair in metric_pairs:
                pair_images = [(metric, file) for metric, file in image_files if metric in pair]
                pair_images.sort(key=lambda x: pair.index(x[0]))  # sort the pair_images according to the order in pair
                if pair_images:
                    f.write('<div class="container">\n')
                    for _, image_file in pair_images:  # unpack the tuple to get the image_file
                        f.write(f'<img src="{image_file}" />\n')
                    f.write('</div>\n')

            f.write('</body>\n</html>\n')

        # Return the path to the index page
        return os.path.realpath(os.path.join(output_dir, "index.html"))

    def __write_csv(self, data, run_id, for_mini_dataset=False):
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

        return file_path

    def generate_graphs_by_run_id(self, run_id, format='html', for_mini_dataset=False):
        data = self.__fetch_metrics(run_id, for_mini_dataset)

        if not data:
            print(f"No metrics found for run: {run_id}")
            return

        if format == 'csv':
            return self.__write_csv(data, run_id, for_mini_dataset), data
        else:
            return self.__plot_graphs(data, run_id, for_mini_dataset), data

    def generate_index_page(self, processed_runs):
        # generate an index page
        index_page = f"{self.PROJECT_ROOT}/graphs/index.html"

        with open(index_page, 'w') as f:
            f.write('''
            <html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
                <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"></script>
                <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"></script>
                <script>
    $(document).ready(function() {
        $(".utc-timestamp").each(function() {
            var utc_timestamp = $(this).text();
            var date = new Date(utc_timestamp + "Z");
            var local_timestamp = date.getFullYear() + "-" +
                                  ("0" + (date.getMonth()+1)).slice(-2) + "-" +
                                  ("0" + date.getDate()).slice(-2) + " " +
                                  ("0" + date.getHours()).slice(-2) + ":" +
                                  ("0" + date.getMinutes()).slice(-2) + ":" +
                                  ("0" + date.getSeconds()).slice(-2);
            $(this).text(local_timestamp);
        });
    });
                </script>
                <style>
                    body {
                        background-color: #343a40;
                        color: #fff;
                    }
                    .card {
                        background-color: #343a40;
                        color: #fff;
                    }
                    .card table {
                        background-color: #343a40;
                    }
                </style>
            </head>
            <body>
                <table class="table table-dark">
                    <thead>
                        <tr>
                            <th>Run ID</th>
                            <th>Timestamp</th>
                            <th>Status</th>
                            <th>Num Epochs</th>
                            <th>Accuracy</th>
                            <th>Top 5 Accuracy</th>
                            <th>Config</th>
                        </tr>
                    </thead>
                    <tbody>
            ''')

            for idx, (run, outfile, metrics) in enumerate(processed_runs):
                last_epoch_metrics = metrics[max(metrics.keys())]
                config_data = json.loads(run.get('config', '{}')) # convert the json string to a dictionary
                accuracy = last_epoch_metrics['validation_accuracy']
                top_5_accuracy = last_epoch_metrics['validation_top_5_accuracy']

                # Parse and reformat timestamp
                timestamp = datetime.strptime(run.get("started_at", "NA"), "%Y-%m-%dT%H:%M:%S.%f")
                formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                f.write(f'''
                    <tr>
                        <td><a href="{pathname2url(outfile)}" style="color: #17a2b8;">{run["run_id"]}</a></td>
                        <td class="utc-timestamp">{formatted_timestamp}</td>
                        <td>{run.get("run_status", "None")}</td>
                        <td>{len(metrics)}</td>
                        <td>{accuracy}</td>
                        <td>{top_5_accuracy}</td>
                        <td>
                            <button class="btn btn-primary" type="button" data-toggle="collapse" data-target="#config{idx}" aria-expanded="false" aria-controls="config{idx}">
                                Show/Hide Config
                            </button>
                        </td>
                    </tr>
                    <tr class="collapse" id="config{idx}">
                        <td colspan="6">
                            <div class="card card-body">
                                <div class="row">
                ''')

                # Iterate through the config data and create a grid of key-value pairs
                for i, (key, value) in enumerate(config_data.items()):
                    f.write(f'''
                    <div class="col-sm-6">
                        <strong>{key}</strong>: {value}
                    </div>
                    ''')

                f.write('''
                                </div>
                            </div>
                        </td>
                    </tr>
                ''')

            f.write('''
                    </tbody>
                </table>
            </body>
            </html>
            ''')

        return index_page

    def batch_generate_html(self, runs, for_mini_dataset=False):
        processed_runs = []

        for run in runs:
            # filter out runs that aren't for the right dataset type
            if for_mini_dataset and run['training_dataset_type'] != 'mini':
                continue
            elif not for_mini_dataset and run['training_dataset_type'] == 'mini':
                continue

            graph_index_page, metrics = self.generate_graphs_by_run_id(run['run_id'], 'html', for_mini_dataset)
            processed_runs.append((run, graph_index_page, metrics))
            print(f"Generated graphs for run: {run['run_id']} to outfile {graph_index_page}")
            print(f"Metrics: {metrics}")

        index_page = self.generate_index_page(processed_runs)
        return index_page