import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from src.metrics.metrics_manager import MetricsManager

def fetch_metrics(model_name):
    metrics_manager = MetricsManager()

    # fetch the training metrics from the training dataset
    training_metrics = metrics_manager.get_metrics('train_model', model_name, 'train')
    validation_metrics = metrics_manager.get_metrics('test_model', model_name, 'validation')

    data = {}

    for item in training_metrics:
        epoch = item['epoch']
        if epoch not in data:
            data[epoch] = {}
        data[epoch]['training_loss'] = float(item['loss'])
        data[epoch]['training_accuracy'] = float(item['accuracy'])
        data[epoch]['training_top_5_acc'] = float(item['top_5_acc'])

    for item in validation_metrics:
        epoch = item['epoch']
        if epoch not in data:
            data[epoch] = {}
        data[epoch]['validation_loss'] = float(item['loss'])
        data[epoch]['validation_accuracy'] = float(item['accuracy'])
        data[epoch]['validation_top_5_acc'] = float(item['top_5_acc'])

    return data

def plot_graph(data):
    plt.style.use('dark_background')

    epochs = sorted(data.keys())

    # initialize subplots in a 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7))


    # plot the accuracy metrics
    ax1.plot(epochs, [data[epoch].get('training_accuracy') for epoch in epochs], linestyle='-', marker='o', color='yellow', label='training accuracy')
    ax1.plot(epochs, [data[epoch].get('validation_accuracy') for epoch in epochs], linestyle='-', marker='o', color='cyan', label='validation accuracy')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Epochs')
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # plot the top_5_accuracy metrics
    ax2.plot(epochs, [data[epoch].get('training_top_5_acc') for epoch in epochs], linestyle='-', marker='o', color='yellow', label='training top 5 accuracy')
    ax2.plot(epochs, [data[epoch].get('validation_top_5_acc') for epoch in epochs], linestyle='-', marker='o', color='cyan', label='validation top 5 accuracy')

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Top 5 Accuracy')
    ax2.set_title('Top 5 Accuracy vs Epochs')
    ax2.legend()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # plot the loss metrics
    ax3.plot(epochs, [data[epoch].get('training_loss') for epoch in epochs], linestyle='-', marker='o', color='yellow', label='training loss')
    ax3.plot(epochs, [data[epoch].get('validation_loss') for epoch in epochs], linestyle='-', marker='o', color='cyan', label='validation loss')

    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.set_title('Loss vs Epochs')
    ax3.legend()
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    # hide the unused subplot
    ax4.axis('off')

    # adjust subplot layout and show plot
    plt.tight_layout()
    plt.show()

def print_csv(data):
    epochs = sorted(data.keys())
    print("epoch,training_loss,training_accuracy,training_top_5_accuracy,validation_loss,validation_accuracy,validation_top_5_acc")
    for epoch in epochs:
        print(f"{epoch},"
                f"{data[epoch].get('training_loss')},"
                f"{data[epoch].get('training_accuracy')},"
                f"{data[epoch].get('training_top_5_acc')},"
                f"{data[epoch].get('validation_loss')},"
                f"{data[epoch].get('validation_accuracy')},"
                f"{data[epoch].get('validation_top_5_acc')}")

def main():
    parser = argparse.ArgumentParser(description='Fetch data from DynamoDB and plot the graph.')
    parser.add_argument('--model-name', required=True, help='Name of the model')
    parser.add_argument('--csv', action='store_true', help='Output data as CSV')
    args = parser.parse_args()

    data = fetch_metrics(args.model_name)
    if args.csv:
        print_csv(data)
    else:
        plot_graph(data)


if __name__ == '__main__':
    main()