import logging
import time

import numpy as np
from multiprocessing import Process
from src.utils.rabbitmq import get_connection, serialize, deserialize
from src.utils.constants import BROADCAST_EXCHANGE, RETRIEVAL_QUEUE
from src.utils.data_loader import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Master(Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n = int(config.get("global", "n"))
        self.iteration_limit = int(config.get("global", "iteration_limit"))
        self.learning_rate = float(config.get("global", "learning_rate"))
        self.A, self.y = load_data(config)
        self.m, self.k = self.A.shape
        self.x = np.ones((self.k, 1))
        logger.info("Master node initialized")

    def run(self):
        logger.info("Master node started")

        connection = get_connection()
        channel = connection.channel()
        
        channel.exchange_declare(exchange=BROADCAST_EXCHANGE, exchange_type='fanout')
        channel.queue_declare(queue=RETRIEVAL_QUEUE)

        logger.info("Waiting for slaves to be ready...")
        time.sleep(5)

        for i in range(self.iteration_limit):
            logger.info("--- Iteration %d/%d ---", i + 1, self.iteration_limit)
            channel.queue_purge(queue=RETRIEVAL_QUEUE)

            # 1. Broadcast current x
            channel.basic_publish(exchange=BROADCAST_EXCHANGE, routing_key='', body=serialize(self.x))

            # 2. Collect A_i*x and compute z = Ax - y
            Ax_parts = self._collect_results(channel)
            Ax = np.vstack(Ax_parts)
            z = Ax - self.y

            # 3. Broadcast z
            channel.basic_publish(exchange=BROADCAST_EXCHANGE, routing_key='', body=serialize(z))

            # 4. Collect A^T_i*z_i and compute full gradient
            grad_parts = self._collect_results(channel)
            gradient = np.sum(grad_parts, axis=0)

            # 5. Update x using gradient descent
            self.x -= self.learning_rate * gradient

            logger.info("Iteration %d complete. Current x:\n%s", i + 1, self.x)

        logger.info("Training finished")
        logger.info("Final model parameters (x):\n%s", self.x)
        connection.close()

    def _collect_results(self, channel):
        """Collects results from all slaves."""
        results = []
        while len(results) < self.n:
            method_frame, _, body = channel.basic_get(queue=RETRIEVAL_QUEUE)
            if method_frame:
                results.append(deserialize(body))
                channel.basic_ack(method_frame.delivery_tag)
            else:
                time.sleep(0.01) # Wait briefly if queue is empty
        
        # Sort results by slave ID to ensure correct order for vstack
        results.sort(key=lambda item: item['id'])
        return [item['data'] for item in results]