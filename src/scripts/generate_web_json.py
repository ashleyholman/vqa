import argparse
from nturl2path import pathname2url
import os
from urllib.parse import urljoin
import webbrowser

from src.metrics.graph_generator import GraphGenerator
from src.util.run_manager import RunManager

def main():
    parser = argparse.ArgumentParser(description='Fetch data from DynamoDB and plot the graph.')
    parser.add_argument('--max-runs', type=int, default=10, help='Maximum number of runs to fetch')
    parser.add_argument('--regenerate-existing', action='store_true', help='Regeneral JSON files for existing runs')

    args = parser.parse_args()

    graph_generator = GraphGenerator()

    run_manager = RunManager()
    recent_runs = run_manager.get_recent_runs(args.max_runs)
    json_files = graph_generator.generate_json_files(recent_runs, args.regenerate_existing)

    print(f'Results written to JSON files to {json_files}.\n')

if __name__ == '__main__':
    main()