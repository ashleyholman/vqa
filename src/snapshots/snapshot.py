class Snapshot:
    def __init__(self, name, model, dataset, optimizer, metadata):
        self.name = name
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.metadata = metadata

    def get_name(self):
        return self.name

    def get_model(self):
        return self.model

    def get_dataset(self):
        return self.dataset

    def get_optimizer(self):
        return self.optimizer

    def get_metadata(self):
        return self.metadata

    def isLightweight(self):
        return self.metadata['lightweight']