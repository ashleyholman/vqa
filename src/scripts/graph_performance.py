import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from src.metrics.metrics_manager import MetricsManager

import webbrowser
from urllib.parse import urljoin
from urllib.request import pathname2url

def fetch_metrics(model_name):
    metrics_manager = MetricsManager('graph-performance')

    # fetch the metrics from the training and validation datasets
    datasets = {
        'training': metrics_manager.get_metrics('train_model', model_name, 'train'),
        'validation': metrics_manager.get_metrics('test_model', model_name, 'validation')
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

def plot_graphs(data, model_name):
    plt.style.use('dark_background')
    epochs = sorted(data.keys())
    metrics = set([k.split('_', 1)[1] for k in data[epochs[0]].keys() if '_' in k])
    output_dir = f"graphs/{model_name}"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    metric_pairs = [('accuracy', 'top_5_accuracy'), 
                    ('precision_macro', 'precision_micro'), 
                    ('recall_macro', 'recall_micro'), 
                    ('f1_score_macro', 'f1_score_micro')]

    # Plot 'loss' first as a larger graph
    if 'loss' in metrics:
        plt.figure(figsize=(10, 5))
        ax = plt.gca()
        ax.plot(epochs, [data[epoch].get(f'training_loss') for epoch in epochs], linestyle='-', marker='o', color='yellow', label='training')
        ax.plot(epochs, [data[epoch].get(f'validation_loss') for epoch in epochs], linestyle='-', marker='o', color='cyan', label='validation')
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
            ax.plot(epochs, [data[epoch].get(f'training_{metric}') for epoch in epochs], linestyle='-', marker='o', color='yellow', label='training')
            ax.plot(epochs, [data[epoch].get(f'validation_{metric}') for epoch in epochs], linestyle='-', marker='o', color='cyan', label='validation')
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
    parser.add_argument('--model-name', required=True, help='Name of the model')
    parser.add_argument('--csv', action='store_true', help='Output data as CSV')
    args = parser.parse_args()

    data = fetch_metrics(args.model_name)
    if args.csv:
        print_csv(data)
    else:
        plot_graphs(data, args.model_name)

if __name__ == '__main__':
    main()

