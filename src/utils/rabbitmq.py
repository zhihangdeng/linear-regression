import pika
import pickle

def get_connection():
    return pika.BlockingConnection(pika.ConnectionParameters('localhost'))

def serialize(data):
    return pickle.dumps(data)

def deserialize(body):
    return pickle.loads(body)