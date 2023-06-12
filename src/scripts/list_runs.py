from src.util.run_manager import RunManager

if __name__ == "__main__":
    run_manager = RunManager()

    runs = run_manager.get_recent_runs(10)
    for run in runs:
        print(run)