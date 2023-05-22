import argparse
import re
from collections import defaultdict
from datetime import datetime

from colorama import Fore, Style

from src.snapshots.vqa_snapshot_manager import VQASnapshotManager

def main():
    parser = argparse.ArgumentParser(description='S3 Snapshot Cleaner.')
    parser.add_argument('--dry-run', action='store_true', help='Simulate the cleanup process.')
    args = parser.parse_args()

    snapshot_manager = VQASnapshotManager()

    snapshots = snapshot_manager.list_snapshots()

    # Group snapshots based on non-timestamp part of the name
    snapshot_groups = defaultdict(list)
    for snapshot in snapshots:
        match = re.match(r'^(.*)_(\d{8}_\d{6})$', snapshot)
        if match:
            base_name, timestamp = match.groups()
            try:
                timestamp_obj = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                snapshot_groups[base_name].append((timestamp_obj, snapshot))
            except ValueError:
                print(f"Warning: Snapshot '{snapshot}' has an invalid timestamp.")
                continue
        else:
            print(f"Warning: Snapshot '{snapshot}' does not conform to expected naming format.  Skipping.")
            continue

    for base_name, group in snapshot_groups.items():
        if len(group) > 1:
            # Sort snapshots by timestamp in descending order
            group.sort(key=lambda x: x[0], reverse=True)

            print(f"{Fore.GREEN}keep  : {group[0][1]}{Style.RESET_ALL}")

            # Skip the first snapshot (the most recent one)
            for _, snapshot in group[1:]:
                print(f"{Fore.RED}delete: {snapshot}{Style.RESET_ALL}")
                if not args.dry_run:
                    snapshot_manager.delete_snapshot(snapshot)
        else:
            print(f"{Fore.GREEN}keep  : {group[0][1]}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
