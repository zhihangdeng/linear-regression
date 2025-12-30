import numpy as np
import logging
from src.code.base import Code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShiftAndAdd(Code):
    def __init__(self, n, k):
        super().__init__()
        if n < k:
            raise ValueError("n must be greater than or equal to k")
        self.n = n
        self.k = k


    def encode(self, data_partitions):
        """
        Encodes k data partitions into n coded partitions.
        data_partitions: A list of k numpy arrays.
        """
        if len(data_partitions) != self.k:
            raise ValueError(f"Expected {self.k} partitions to encode, but got {len(data_partitions)}")

        data_partitions = [np.vstack([partition, -np.sum(partition, axis=0, keepdims=True)]) for partition in data_partitions]

        local_parity_1 = data_partitions[0] + data_partitions[1] + data_partitions[2]
        local_parity_2 = data_partitions[3] + data_partitions[4] + data_partitions[5]

        sum_shift_0 = data_partitions[0] + data_partitions[3]
        sum_shift_1 = data_partitions[1] + data_partitions[4]
        sum_shift_2 = data_partitions[2] + data_partitions[5]

        global_parity = sum_shift_0 + np.roll(sum_shift_1, 1, axis=0) + np.roll(sum_shift_2, 2, axis=0)

        encoded_partitions = data_partitions + [local_parity_1, local_parity_2, global_parity]

        return encoded_partitions


    def decode(self, results):
        """
        Decodes the original k results from any k received results.
        results: A list of k dictionaries, each {'id': slave_id, 'data': result_data}
        """
        if len(results) < self.k:
            raise ValueError(f"Need at least {self.k} results to decode, but got {len(results)}")

        slave_ids = list(results.keys())
        received_data = list(results.values())

        missing_ids = sorted([i for i in range(self.n) if i not in slave_ids])

        if len(missing_ids) == 1:
            if missing_ids[0] == 7:
                results = self._global_repair(missing_ids[0], results)
            elif missing_ids[0] in [0, 1, 2, 6]:
                results = self._local_repair_group_1(missing_ids[0], results)
            else:
                results = self._local_repair_group_2(missing_ids[0], results)
        elif len(missing_ids) == 2:
            if missing_ids[1] == 6:
                if missing_ids[0] in [0, 1, 2]:
                    results = self._global_repair(missing_ids[0], results)
                    results = self._local_repair_group_1(missing_ids[1], results)
                else:
                    results = self._local_repair_group_2(missing_ids[0], results)
                    results = self._local_repair_group_1(missing_ids[1], results)
            elif missing_ids[1] == 7:
                if missing_ids[0] in [3, 4, 5]:
                    results = self._global_repair(missing_ids[0], results)
                    results = self._local_repair_group_2(missing_ids[1], results)
                else:
                    results = self._local_repair_group_1(missing_ids[0], results)
                    results = self._local_repair_group_2(missing_ids[1], results)
            elif missing_ids[1] == 8:
                if missing_ids[0] in [0, 1, 2, 6]:
                    results = self._local_repair_group_1(missing_ids[0], results)
                    results = self._global_repair(missing_ids[1], results)
                else:
                    results = self._local_repair_group_2(missing_ids[0], results)
                    results = self._global_repair(missing_ids[1], results)
            elif missing_ids[0] in [0, 1, 2] and missing_ids[1] in [0, 1, 2]:
                results = self._two_in_one_group_repair_1(missing_ids, results)
            elif missing_ids[0] in [3, 4, 5] and missing_ids[1] in [3, 4, 5]:
                results = self._two_in_one_group_repair_2(missing_ids, results)
            else:
                results = self._local_repair_group_1(missing_ids[0], results)
                results = self._local_repair_group_2(missing_ids[1], results)
        elif len(missing_ids) == 3:
            if missing_ids[0] in [0, 1, 2] and missing_ids[1] in [0, 1, 2] and missing_ids[2] in [3, 4, 5, 7]:
                results = self._two_in_one_group_repair_1(missing_ids[:2], results)
                results = self._local_repair_group_2(missing_ids[2], results)
            elif missing_ids[0] in [3, 4, 5] and missing_ids[1] in [3, 4, 5] and missing_ids[2] in [0, 1, 2, 6]:
                results = self._two_in_one_group_repair_2(missing_ids[:2], results)
                results = self._local_repair_group_1(missing_ids[2], results)
            else:
                raise ValueError("Cannot repair more than two failures in different groups.")
        
        original_partitions = [results[i][:-1] for i in range(self.k)]
        return original_partitions


    def _local_repair_group_1(self, missing_id, results):
        if missing_id == 6:
            repaired = results[0] + results[1] + results[2]
        else:
            repaired = results[6] - sum(results[i] for i in range(3) if i != missing_id)
        results[missing_id] = repaired
        return results


    def _local_repair_group_2(self, missing_id, results):
        if missing_id == 7:
            repaired = results[3] + results[4] + results[5]
        else:
            repaired = results[7] - sum(results[i] for i in range(3, 6) if i != missing_id)
        results[missing_id] = repaired
        return results


    def _global_repair(self, missing_id, results):
        if missing_id == 8:
            sum_shift_0 = results[0] + results[3]
            sum_shift_1 = results[1] + results[4]
            sum_shift_2 = results[2] + results[5]
            repaired = sum_shift_0 + np.roll(sum_shift_1, 1, axis=0) + np.roll(sum_shift_2, 2, axis=0)
        else:
            repaired = results[8].copy()
            if missing_id in [0, 3]:
                sum_shift_1 = results[1] + results[4]
                sum_shift_2 = results[2] + results[5]
                repaired -= np.roll(sum_shift_1, 1, axis=0) + np.roll(sum_shift_2, 2, axis=0)
                repaired -= results[3] if missing_id == 0 else results[0]
            elif missing_id in [1, 4]:
                sum_shift_0 = results[0] + results[3]
                sum_shift_2 = results[2] + results[5]
                repaired -= sum_shift_0 + np.roll(sum_shift_2, 2, axis=0)
                repaired -= np.roll(results[4], 1, axis=0) if missing_id == 1 else np.roll(results[1], 1, axis=0)
                repaired = np.roll(repaired, -1, axis=0)
            else:
                sum_shift_0 = results[0] + results[3]
                sum_shift_1 = results[1] + results[4]
                repaired -= sum_shift_0 + np.roll(sum_shift_1, 1, axis=0)
                repaired -= np.roll(results[5], 2, axis=0) if missing_id == 2 else np.roll(results[2], 2, axis=0)
                repaired = np.roll(repaired, -2, axis=0)
        results[missing_id] = repaired
        return results


    def _two_in_one_group_repair_1(self, missing_ids, results):
        p = results[6] - sum(results[i] for i in range(3) if i not in missing_ids)
        q = results[8] - sum(np.roll(results[i + 3], i, axis=0) for i in range(3) if i in missing_ids) - sum(np.roll(results[i] + results[i + 3], i, axis=0) for i in range(3) if i not in missing_ids)

        delta = missing_ids[1] - missing_ids[0]
        r = q - np.roll(p, missing_ids[0], axis=0)
        s = -np.roll(r, -missing_ids[0], axis=0)

        results = self._invert_one_minus_z_delta(s, delta, missing_ids[1], results)
        results[missing_ids[0]] = p - results[missing_ids[1]]

        return results


    def _two_in_one_group_repair_2(self, missing_ids, results):
        p = results[7] - sum(results[i] for i in range(3, 6) if i not in missing_ids)
        q = results[8] - sum(np.roll(results[i - 3], i - 3, axis=0) for i in range(3, 6) if i in missing_ids) - sum(np.roll(results[i] + results[i - 3], i - 3, axis=0) for i in range(3, 6) if i not in missing_ids)

        delta = missing_ids[1] - missing_ids[0]
        r = q - np.roll(p, missing_ids[0] - 3, axis=0)
        s = -np.roll(r, -(missing_ids[0] - 3), axis=0)

        results = self._invert_one_minus_z_delta(s, delta, missing_ids[1], results)
        results[missing_ids[0]] = p - results[missing_ids[1]]

        return results


    def _invert_one_minus_z_delta(self, s, delta, missing_id, results):
        n = s.shape[0]
        repaired = np.zeros_like(s)
        for i in range(1, n):
            u = (i * delta) % n
            v = ((i - 1) * delta) % n
            repaired[u] = s[u] + repaired[v]

        phi = np.sum(repaired, axis=0, keepdims=True)
        c = -phi / n
        repaired += c
        results[missing_id] = repaired
        return results

