from src.util.run_manager import RunManager

if __name__ == "__main__":
    run_manager = RunManager()

    runs = run_manager.get_recent_runs(10)
    for run in runs:
        # Extract run_id by splitting 'PK' at ':' and getting the second part
        run_id = run['PK'].split(':')[1]
        
        # Fetch the actual primary run record using run_id
        run_record = run_manager.get_run(run_id)
        
        print(run_record)