import numpy as np
import logging
from src.code.base import Code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RS(Code):
    def __init__(self, n, k):
        super().__init__()
        if n < k:
            raise ValueError("n must be greater than or equal to k")
        self.n = n
        self.k = k
        
        # Generate the n x k Vandermonde matrix for encoding
        self.encoding_matrix = np.vander(np.arange(1, self.n + 1), self.k, increasing=True)

    def encode(self, data_partitions):
        """
        Encodes k data partitions into n coded partitions.
        data_partitions: A list of k numpy arrays.
        """
        if len(data_partitions) != self.k:
            raise ValueError(f"Expected {self.k} partitions to encode, but got {len(data_partitions)}")

        base_shape = data_partitions[0].shape
        encoded_partitions = [np.zeros(base_shape, dtype=data_partitions[0].dtype) for _ in range(self.n)]

        for i in range(self.n):
            for j in range(self.k):
                encoded_partitions[i] += self.encoding_matrix[i, j] * data_partitions[j]
        
        return encoded_partitions

    def decode(self, results):
        """
        Decodes the original k results from any k received results.
        results: A list of k dictionaries, each {'id': slave_id, 'data': result_data}
        """
        if len(results) < self.k:
            raise ValueError(f"Need at least {self.k} results to decode, but got {len(results)}")

        # Get the slave IDs and the data from the fastest k slaves
        slave_ids = [res['id'] for res in results]
        received_data = [res['data'] for res in results]

        # Construct the k x k decoding matrix from the corresponding rows of the original encoding matrix
        decoding_matrix = self.encoding_matrix[slave_ids, :]

        # We need to solve a system of linear equations.
        # To do this efficiently for matrices, we flatten them, solve, and then reshape.
        sample_shape = received_data[0].shape
        # Stack the flattened received data into a matrix (k x flattened_len)
        b = np.array([d.flatten() for d in received_data])

        # Solve G_k * X = b, where X is the matrix of flattened original data
        original_flattened = np.linalg.solve(decoding_matrix, b)

        # Reshape the solved flattened data back to the original partition shape
        original_partitions = [v.reshape(sample_shape) for v in original_flattened]

        return original_partitions