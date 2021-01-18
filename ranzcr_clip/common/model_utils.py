import yaml


class ModelConfig:
    def __init__(self, config_file: str):
        self.params = yaml.safe_load(open(config_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)
