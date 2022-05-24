# from kafka import KafkaProducer
# import json

# def main():
#     producer = KafkaProducer(
#         bootstrap_servers='127.0.0.1:9092',
#         value_serializer=lambda v: json.dumps(v).encode('utf-8'))
#     producer.send('pytest', {'message': 'Hello kafka'})
#     producer.close()

# if __name__ == '__main__':
#     main()

from kafka import KafkaProducer
from kafka.errors import KafkaError

producer = KafkaProducer(bootstrap_servers=['127.0.0.1:9092'])
topic = "kafkatopic"

producer.send(topic, b'test message')