from pymongo import MongoClient
from flask import Flask, request, jsonify
from kafka import KafkaProducer
from flask_cors import CORS
import json, time

app = Flask(__name__)
CORS(app)

# ── MongoDB on localhost (Docker port 27017 mapped to host) ───────
client = MongoClient("mongodb://localhost:27017/")
db = client["game_db"]

# ── Kafka on localhost ────────────────────────────────────────────
producer = None
while producer is None:
    try:
        producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        print("✅ Kafka connected")
    except Exception as e:
        print(f"⏳ Kafka not ready: {e}")
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
    if data["player_id"] is None or data["score"] is None:
        return jsonify({"error": "Missing required fields"}), 400

    producer.send('game_topic', data)
    producer.flush()
    return jsonify({"status": "sent", "data": data})


@app.route('/api/ml-results', methods=['GET'])
def ml_results():
    return jsonify(list(db.players_classified.find({}, {"_id": 0})))

@app.route('/api/fuzzy-results', methods=['GET'])
def fuzzy_results():
    return jsonify(list(db.fuzzy_results.find({}, {"_id": 0})))

@app.route('/api/metrics', methods=['GET'])
def metrics():
    return jsonify(db.model_metrics.find_one({}, {"_id": 0}))

@app.route('/api/game-evaluation', methods=['GET'])
def game_evaluation():
    data = db.game_evaluation.find_one({}, {"_id": 0})
    if not data:
        return jsonify({"error": "No evaluation yet"}), 404
    return jsonify(data)

@app.route('/')
def home():
    return "Backend running on localhost:5000"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)