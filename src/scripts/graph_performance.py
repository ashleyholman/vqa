import os
import argparse
import matplotlib.pyplot as plt
import webbrowser

from matplotlib.ticker import MaxNLocator
from urllib.parse import urljoin
from urllib.request import pathname2url
from src.metrics.graph_generator import GraphGenerator

from src.metrics.metrics_manager import MetricsManager
from src.models.model_configuration import ModelConfiguration

def fetch_metrics(args):
    metrics_manager = MetricsManager('graph-performance')

    if args.run_id:
        if args.mini_dataset:
            datasets = {'mini': metrics_manager.get_metrics_by_run_id(args.run_id, 'mini')}
        else:
            datasets = {
                'training': metrics_manager.get_metrics_by_run_id(args.run_id, 'train'),
                'validation': metrics_manager.get_metrics_by_run_id(args.run_id, 'validation')
            }
    else:
        datasets = {
            'training': metrics_manager.get_metrics('train_model', args.model_name, 'train'),
            'validation': metrics_manager.get_metrics('test_model', args.model_name, 'validation')
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

def plot_graphs(data, args):
    plt.style.use('dark_background')
    epochs = sorted(data.keys())
    metrics = set([k.split('_', 1)[1] for k in data[epochs[0]].keys() if '_' in k])

    output_dir_suffix = "_mini" if args.mini_dataset else ""
    output_name = f"run_{args.run_id}" if args.run_id else args.model_name
    output_dir = f"graphs/{output_name}{output_dir_suffix}"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    metric_pairs = [('accuracy', 'top_5_accuracy'), 
                    ('precision_macro', 'precision_micro'), 
                    ('recall_macro', 'recall_micro'), 
                    ('f1_score_macro', 'f1_score_micro')]

    # Determine the datasets to plot metrics for, and their corresponding colours
    dataset_prefixes = {'mini': 'pink'} if args.mini_dataset else {'training': 'yellow', 'validation': 'cyan'}

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

    # Open the HTML file in default browser
    webbrowser.open('file://' + os.path.realpath(os.path.join(output_dir, "index.html")))

def print_csv(data):
    header = ['epoch', 'training_loss']

    # Collect all unique keys from the data
    for epoch in data.keys():
        for key in data[epoch].keys():
            if key not in header:
                header.append(key)

    print(','.join(header))

    for epoch in sorted(data.keys()):
        row = [str(data[epoch].get(key, '')) for key in header]
        print(','.join(row))

def main():
    parser = argparse.ArgumentParser(description='Fetch data from DynamoDB and plot the graph.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--latest', action='store_true', help='Report on the latest run')
    group.add_argument('--run-id', help='ID of the run to report on')

    parser.add_argument('--csv', action='store_true', help='Output data as CSV')
    parser.add_argument('--mini-dataset', action='store_true', help='Output data for the mini dataset')

    args = parser.parse_args()

    #if args.latest:
        # obtain the latest run ID
        # TODO: implement

    graph_generator = GraphGenerator()

    if args.csv:
        format = 'csv'
    else:
        format = 'html'
    
    graph_generator.generate_graphs_by_run_id(args.run_id, format, args.mini_dataset)

if __name__ == '__main__':
    main()