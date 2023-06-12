import argparse
from nturl2path import pathname2url
import os
from urllib.parse import urljoin
import webbrowser

from src.metrics.graph_generator import GraphGenerator
from src.util.run_manager import RunManager

def main():
    parser = argparse.ArgumentParser(description='Fetch data from DynamoDB and plot the graph.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--latest', action='store_true', help='Report on the latest run')
    group.add_argument('--run-id', help='ID of the run to report on')
    group.add_argument('--all', action='store_true', help='Report on all recent runs')

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

    if args.all:
        run_manager = RunManager()
        recent_runs = run_manager.get_recent_runs(10)
        if format == 'html':
            index_page = graph_generator.batch_generate_html(recent_runs, args.mini_dataset)
            webbrowser.open_new_tab(urljoin('file:', pathname2url(index_page)))
    else:
        outfile = graph_generator.generate_graphs_by_run_id(args.run_id, format, args.mini_dataset)
        if format == 'html':
            webbrowser.open_new_tab(urljoin('file:', pathname2url(outfile)))
        else:
            print(f'Results written to CSV to {outfile}.\n')
            # print the CSV to stdout
            with open(outfile, 'r') as f:
                print(f.read())

if __name__ == '__main__':
    main()