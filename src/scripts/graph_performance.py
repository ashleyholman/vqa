import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

from src.metrics.metrics_manager import MetricsManager


def fetch_metrics(model_name, dataset_type):
    metrics_manager = MetricsManager()
    performance_metrics = metrics_manager.get_performance_metrics(model_name, dataset_type)
    training_metrics = metrics_manager.get_training_metrics(model_name, dataset_type)

    data = defaultdict(lambda: {'accuracy': None, 'top_5_acc': None, 'loss': None})

    for item in performance_metrics:
        epoch = item['epoch']
        data[epoch]['accuracy'] = float(item['accuracy'])
        data[epoch]['top_5_acc'] = float(item['top_5_acc'])

    for item in training_metrics:
        epoch = item['epoch']
        data[epoch]['loss'] = float(item['loss'])

    return data


def plot_graph(data):
    epochs = sorted(data.keys())
    accuracy = [data[epoch]['accuracy'] for epoch in epochs]
    top_5_acc = [data[epoch]['top_5_acc'] for epoch in epochs]
    loss = [data[epoch]['loss'] for epoch in epochs]

    fig, ax = plt.subplots()

    ax.plot(epochs, accuracy, color="blue", marker="o", label='Accuracy')
    ax.plot(epochs, top_5_acc, color="green", marker="o", label='Top 5 Accuracy')

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Accuracy / Top 5 Accuracy", color="blue", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(epochs, loss, color="red", marker="o", label='Loss')
    ax2.set_ylabel("Loss", color="red", fontsize=14)

    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Fetch data from DynamoDB and plot the graph.')
    parser.add_argument('--model-name', required=True, help='Name of the model')
    parser.add_argument('--dataset-type', required=True, help='Type of the dataset')
    args = parser.parse_args()

    data = fetch_metrics(args.model_name, args.dataset_type)
    plot_graph(data)


if __name__ == '__main__':
    main()