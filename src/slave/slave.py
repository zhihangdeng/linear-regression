import logging
import time
import os
import pandas as pd
import numpy as np
from multiprocessing import Process
from src.utils.rabbitmq import get_connection, serialize, deserialize
from src.utils.constants import BROADCAST_EXCHANGE, RETRIEVAL_QUEUE, DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(DATA_DIR, "data.csv")

class Slave(Process):
    def __init__(self, config, id):
        super().__init__()
        self.id = id
        self.config = config
        self.k = int(config.get("global", "k"))
        self.n = int(config.get("global", "n"))
        self.iteration_limit = int(config.get("global", "iteration_limit"))
    
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
        df = pd.read_csv(DATA_PATH)
        A = df.iloc[:, :-1].values

        rows, cols = A.shape
        if rows % self.k != 0:
            rows_to_add = self.k - (rows % self.k)        
            A = np.vstack([A, np.zeros((rows_to_add, cols))])
        rows, cols = A.shape
        if (cols + 1) % self.k != 0:
            cols_to_add = self.k - ((cols + 1) % self.k)
            A = np.hstack([A, np.zeros((rows, cols_to_add))])
        A = np.hstack([A, -np.sum(A, axis=1, keepdims=True)])
        rows, cols = A.shape

        A_partitions = np.array_split(A, self.k, axis=0)
        AT_partitions = np.array_split(A.T, self.k, axis=0)

        # Generate all n encoded partitions
        from src.code.shift_and_add import ShiftAndAdd
        self.code = ShiftAndAdd(n=self.n, k=self.k)
        encoded_A = self.code.encode(A_partitions)
        encoded_AT = self.code.encode(AT_partitions)

        # Keep only the partition corresponding to this slave's ID
        self.A_i = encoded_A[self.id]
        self.AT_i = encoded_AT[self.id]

        logger.info("Slave node initialized")


    def run(self):
        logger.info("Slave node %d started", self.id)
        
        connection = get_connection()
        channel = connection.channel()

        channel.exchange_declare(exchange=BROADCAST_EXCHANGE, exchange_type='fanout')
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        channel.queue_bind(exchange=BROADCAST_EXCHANGE, queue=queue_name)

        
        for i in range(self.iteration_limit):
            # 1. Receive x and compute A_i*x
            x = self._wait_for_broadcast(channel, queue_name)
            Ax_i = self.A_i @ x
            self._send_result(channel, i, 'ax', Ax_i)

            # 2. Receive z and compute A^T_i*z
            z = self._wait_for_broadcast(channel, queue_name)
            grad_i = self.AT_i @ z
            self._send_result(channel, i, 'grad', grad_i)
        
        logger.info("Slave %d finished processing.", self.id)
        connection.close()

    def _wait_for_broadcast(self, channel, queue_name):
        """Waits for one message from the broadcast queue and returns it."""
        body = None
        while body is None:
            method_frame, _, body = channel.basic_get(queue=queue_name)
            if method_frame:
                channel.basic_ack(method_frame.delivery_tag)
                return deserialize(body)

    def _send_result(self, channel, it, step, data):
        """Sends a result back to the master's retrieval queue."""
        payload = {'it': it, 'id': self.id, 'step': step, 'data': data}
        channel.basic_publish(
            exchange='',
            routing_key=RETRIEVAL_QUEUE,
            body=serialize(payload)
        )