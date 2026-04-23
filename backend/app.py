from flask import Flask, request, jsonify
from kafka import KafkaProducer
import json
import time

app = Flask(__name__)

# Connexion à Kafka (nom du service Docker)
producer = None

while producer is None:
    try:
        producer = KafkaProducer(
            bootstrap_servers='kafka:9092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        print("✅ Connected to Kafka")
    except:
        print("⏳ Kafka not ready, retrying...")
        time.sleep(5)

@app.route('/send', methods=['POST'])
def send_data():
    data = request.json
    
    producer.send('game_topic', data)
    
    return jsonify({"status": "sent to Kafka", "data": data})

@app.route('/')
def home():
    return "Backend running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
