import logging
import time
import os
import pandas as pd
import numpy as np
np.set_printoptions(precision=6, suppress=True) 

from multiprocessing import Process
from src.utils.rabbitmq import get_connection, serialize, deserialize
from src.utils.constants import BROADCAST_EXCHANGE, RETRIEVAL_QUEUE, DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(DATA_DIR, "data.csv")


class Master(Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n = int(config.get("global", "n"))
        self.k = int(config.get("global", "k"))
        self.iteration_limit = int(config.get("global", "iteration_limit"))
        self.learning_rate = float(config.get("global", "learning_rate"))

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

        self.y = df.iloc[:, -1:].values

        self.x = np.zeros((cols, 1))

        from src.code.shift_and_add import ShiftAndAdd
        self.code = ShiftAndAdd(n=self.n, k=self.k)

        logger.info("Master node initialized")


    def run(self):
        logger.info("Master node started")

        connection = get_connection()
        channel = connection.channel()
        
        channel.exchange_declare(exchange=BROADCAST_EXCHANGE, exchange_type='fanout')
        channel.queue_declare(queue=RETRIEVAL_QUEUE)
        channel.queue_purge(queue=RETRIEVAL_QUEUE)

        logger.info("Waiting for slaves to be ready...")
        time.sleep(5)

        for i in range(self.iteration_limit):
            logger.info("--- Iteration %d/%d ---", i + 1, self.iteration_limit)

            # 1. Broadcast current x
            channel.basic_publish(exchange=BROADCAST_EXCHANGE, routing_key='', body=serialize(self.x))

            # 2. Collect A_i*x and compute z = Ax - y
            results = self._collect_k_results(channel, i, 'ax')
            print(results.keys())
            print(results[1].shape)
            Ax_parts = self.code.decode(results)
            Ax = np.vstack(Ax_parts)

            z = Ax - self.y

            # 3. Broadcast z
            channel.basic_publish(exchange=BROADCAST_EXCHANGE, routing_key='', body=serialize(z))

            # 4. Collect A^T_i*z and compute gradient
            results = self._collect_k_results(channel, i, 'grad')
            grad_parts = self.code.decode(results)
            gradient = np.vstack(grad_parts)

            # 5. Update x using gradient descent
            self.x -= self.learning_rate * gradient

            logger.info("Iteration %d complete. \n"
                        "Current x:\n%s", i + 1, self.x)

        logger.info("Training finished")
        logger.info("Final model parameters (x):\n%s", self.x)
        connection.close()


    def _collect_k_results(self, channel, it, step):
        """Collects results from all slaves."""
        results = {}
        while len(results) < self.k + 1:
            method_frame, _, body = channel.basic_get(queue=RETRIEVAL_QUEUE)
            if method_frame:
                result = deserialize(body)
                if result['it'] != it or result['step'] != step:
                    continue  # Ignore results from previous iterations
                results[result['id']] = result['data']
                channel.basic_ack(method_frame.delivery_tag)

        return results