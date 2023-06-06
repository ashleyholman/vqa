import uuid
import datetime
from src.util.dynamodb_helper import DynamoDBHelper


class RunManager:
    def __init__(self):
        self.ddb_helper = DynamoDBHelper()

    def get_unfinished_run(self, state_hash):
        pk = f"unfinished-run:{state_hash}"
        unfinished_runs = self.ddb_helper.query(pk)
        return unfinished_runs[0] if unfinished_runs else None

    def create_run(self, training_dataset_type, validation_dataset_type, max_epochs, state_hash, config):
        run_id = str(uuid.uuid4())
        created_at_timestamp = datetime.datetime.now().isoformat()
        
        # Create the "run" record in DDB
        run_record = {
            'run_id': run_id,
            'training_dataset_type': training_dataset_type,
            'validation_dataset_type': validation_dataset_type,
            'max_epochs': max_epochs,
            'state_hash': state_hash,
            'config': config.to_json_string(),
            'started_at': created_at_timestamp,
            'run_status': 'IN_PROGRESS',
            'GSI_PK': 'run',
            'GSI_SK': f"{created_at_timestamp}#{run_id}"
        }
        self.ddb_helper.put_item(f"run:{run_id}", '0', run_record)

        # Create the "unfinished-run" record in DDB
        unfinished_run_record = {
            'run_id': run_id,
            'state_hash': state_hash,
            'trained_until_epoch': 0,
            'snapshot_name': None,
        }
        self.ddb_helper.put_item(f"unfinished-run:{state_hash}", run_id, unfinished_run_record)

        return run_id

    def update_run(self, run_id, attributes):
        self.ddb_helper.update_item(f"run:{run_id}", '0', attributes)

    def delete_unfinished_run(self, state_hash, run_id):
        self.ddb_helper.delete_item(f"unfinished-run:{state_hash}", run_id)
