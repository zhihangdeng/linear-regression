import configparser
import os

from src.utils.constants import CONFIG_DIR

CONFIG_PATH = os.path.join(CONFIG_DIR, "linear_regression.conf")

class Config:
    def __init__(self, config_path=CONFIG_PATH):
        self.config = configparser.ConfigParser()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        self.config_path = config_path
        self.config.read(self.config_path)

    def get(self, section, option):
        return self.config.get(section, option)