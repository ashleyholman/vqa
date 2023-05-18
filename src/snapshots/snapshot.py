class Snapshot:
    def __init__(self, model, dataset, metadata):
        self.model = model
        self.dataset = dataset
        self.metadata = metadata

    def get_model(self):
        return self.model

    def get_dataset(self):
        return self.dataset

    def get_metadata(self):
        return self.metadata