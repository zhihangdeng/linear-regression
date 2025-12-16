import os

BROADCAST_EXCHANGE = 'broadcast'
RETRIEVAL_QUEUE = 'retrieval'

ROOT_DIR = ""

CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
LOG_DIR = ROOT_DIR
DATA_DIR = os.path.join(ROOT_DIR, "data")