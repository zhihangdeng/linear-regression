import logging
import time
import numpy as np

from multiprocessing import Process
from src.utils.rabbitmq import get_connection, serialize, deserialize
from src.utils.constants import BROADCAST_EXCHANGE, RETRIEVAL_QUEUE
from src.utils.data_loader import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Slave(Process):
    def __init__(self, config, id):
        super().__init__()
        self.id = id
        self.config = config
        self.n = int(config.get("global", "n"))
        self.iteration_limit = int(config.get("global", "iteration_limit"))
        self.A_i, self.AT_i = load_data(config, self.id)
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
            self._send_result(channel, Ax_i)

            # 2. Receive z and compute A^T_i*z_i
            z = self._wait_for_broadcast(channel, queue_name)
            # Each slave gets its corresponding slice of z
            z_i = np.array_split(z, self.n)[self.id]
            grad_i = self.AT_i @ z_i
            self._send_result(channel, grad_i)
        
        logger.info("Slave %d finished processing.", self.id)
        connection.close()

    def _wait_for_broadcast(self, channel, queue_name):
        """Waits for one message from the broadcast queue and returns it."""
        body = None
        while body is None:
            method_frame, _, body = channel.basic_get(queue=queue_name)
            if body is None:
                time.sleep(0.01)
        channel.basic_ack(method_frame.delivery_tag)
        return deserialize(body)

    def _send_result(self, channel, data):
        """Sends a result back to the master's retrieval queue."""
        payload = {'id': self.id, 'data': data}
        channel.basic_publish(
            exchange='',
            routing_key=RETRIEVAL_QUEUE,
            body=serialize(payload)
        )