from kafka import KafkaProducer
import json
import time

while True:
    try:
        producer = KafkaProducer(
            bootstrap_servers='kafka:9092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        print("Connected to Kafka ")
        break
    except:
        print("Kafka not ready, retrying...")
        time.sleep(5)

while True:
    data = {
        "player_id": 1,
        "score": 80,
        "reaction_time": 2.5
    }

    producer.send('game_topic', data)
    print("sent:", data)

    time.sleep(2)
