from pymongo import MongoClient
from flask import Flask, request, jsonify
from kafka import KafkaProducer
import json
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb://mongodb:27017/")
db = client["game_db"]

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
    req = request.json
    data = {
        "player_id"    : req.get("player_id"),
        "score"        : req.get("score"),
        "time"         : req.get("time"),
        "clicks"       : req.get("clicks", 0),
        "moves"        : req.get("moves", 0),
        "errors"       : req.get("errors", 0),
        "response_time": req.get("response_time", 0.0),
        "level"        : req.get("level", 1),
        "success"      : req.get("success", 0),
        "repetition"   : req.get("repetition", 0),
        "timestamp"    : time.time()
    }

    if data["player_id"] is None or data["score"] is None or data["time"] is None:
        return jsonify({"error": "Missing required fields"}), 400

    producer.send('game_topic', data)
    producer.flush()
    return jsonify({"status": "sent to Kafka", "data": data})


@app.route('/api/ml-results', methods=['GET'])
def ml_results():
    data = list(db.players_classified.find({}, {"_id": 0}))
    return jsonify(data)


@app.route('/api/fuzzy-results', methods=['GET'])
def fuzzy_results():
    data = list(db.fuzzy_results.find({}, {"_id": 0}))
    return jsonify(data)


@app.route('/api/metrics', methods=['GET'])
def metrics():
    data = db.model_metrics.find_one({}, {"_id": 0})
    return jsonify(data)


# ─────────────────────────────────────────────
# Game-level evaluation (aggregated Fuzzy-AHP)
# ─────────────────────────────────────────────
@app.route('/api/game-evaluation', methods=['GET'])
def game_evaluation():
    data = db.game_evaluation.find_one({}, {"_id": 0})
    if data is None:
        return jsonify({"error": "No evaluation available yet"}), 404
    return jsonify(data)


@app.route('/')
def home():
    return "Backend running"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
