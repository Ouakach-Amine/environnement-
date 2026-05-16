"""
Backend Flask — Temps réel
Collecte toutes les données, écrit dans MongoDB,
déclenche ML (pkl) + Fuzzy AHP en background thread.
"""
import os, sys, time, threading, subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient

app  = Flask(__name__)
CORS(app)

# ── MongoDB ───────────────────────────────────────────────────────
client = MongoClient("mongodb://localhost:27017/")
db     = client["game_db"]

# ── Chemins vers les scripts ML ───────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE, "..", "ml-service")
PYTHON = sys.executable
_lock  = threading.Lock()

def run_pipeline():
    """Lance model_once.py puis fuzzy_ahp.py après chaque nouveau joueur."""
    if not _lock.acquire(blocking=False):
        return                          # déjà en cours, on skip
    try:
        n = db.players.count_documents({})
        if n < 3:
            print(f"⏳ {n} joueur(s) — min 3 requis pour ML")
            return

        for script in ["model_once.py", "fuzzy_ahp.py"]:
            path = os.path.join(ML_DIR, script)
            r = subprocess.run(
                [PYTHON, path],
                capture_output=True, text=True, timeout=90
            )
            out = r.stdout.strip()
            err = r.stderr.strip()
            if out: print(f"[{script}]", out)
            if err: print(f"[{script} ERR]", err)

        print("✅ Pipeline ML + Fuzzy AHP terminé")
    except Exception as e:
        print(f"❌ Pipeline erreur: {e}")
    finally:
        _lock.release()


# ═══════════════════════════════════════════════════════════════════
# ENDPOINT COLLECTE — toutes les données du joueur
# ═══════════════════════════════════════════════════════════════════
@app.route('/send', methods=['POST'])
def send_data():
    req = request.json or {}

    if req.get("player_id") is None:
        return jsonify({"error": "player_id requis"}), 400

    # ── Anciennes données (pipeline Kafka/Spark déjà existant) ────
    data = {
        "player_id"     : req["player_id"],
        "score"         : req.get("score", 0),
        "time"          : req.get("time", 0),
        "clicks"        : req.get("clicks", 0),
        "moves"         : req.get("moves", 0),
        "errors"        : req.get("errors", 0),
        "response_time" : req.get("response_time", 0.0),
        "level"         : req.get("level", 1),
        "success"       : req.get("success", 0),
        "repetition"    : req.get("repetition", 0),

        # ── Nouvelles données — Pedagogical Dimension ──────────────
        "hints_used"           : req.get("hints_used", 0),
        "correct_answers"      : req.get("correct_answers", 0),
        "wrong_answers"        : req.get("wrong_answers", 0),
        "objectives_completed" : req.get("objectives_completed", 0),
        "knowledge_score"      : req.get("knowledge_score", 0.0),
        "progression_rate"     : req.get("progression_rate", 0.0),
        "retry_after_fail"     : req.get("retry_after_fail", 0),

        # ── Nouvelles données — Technological Dimension ────────────
        "load_time"     : req.get("load_time", 0.0),
        "crash_count"   : req.get("crash_count", 0),
        "lag_events"    : req.get("lag_events", 0),
        "frame_drops"   : req.get("frame_drops", 0),
        "api_errors"    : req.get("api_errors", 0),
        "screen_width"  : req.get("screen_width", 1920),
        "screen_height" : req.get("screen_height", 1080),
        "device_type"   : req.get("device_type", "desktop"),  # desktop/mobile/tablet

        # ── Nouvelles données — Ludic Dimension ────────────────────
        "playtime_voluntary"   : req.get("playtime_voluntary", 0),
        "bonus_collected"      : req.get("bonus_collected", 0),
        "challenges_attempted" : req.get("challenges_attempted", 0),
        "idle_time"            : req.get("idle_time", 0.0),
        "exploration_rate"     : req.get("exploration_rate", 0.0),
        "combo_count"          : req.get("combo_count", 0),
        "skip_count"           : req.get("skip_count", 0),

        # ── Nouvelles données — Behavioural Dimension ──────────────
        "help_requests"      : req.get("help_requests", 0),
        "give_up_count"      : req.get("give_up_count", 0),
        "pause_count"        : req.get("pause_count", 0),
        "total_pause_time"   : req.get("total_pause_time", 0.0),
        "focus_time"         : req.get("focus_time", 0.0),
        "frustration_events" : req.get("frustration_events", 0),
        "session_count"      : req.get("session_count", 1),
        "days_active"        : req.get("days_active", 1),

        "timestamp": time.time()
    }

    # ── Écriture directe MongoDB (upsert) ─────────────────────────
    db.players.update_one(
        {"player_id": data["player_id"]},
        {"$set": data},
        upsert=True
    )
    print(f"✅ Joueur #{data['player_id']} sauvegardé")

    # ── Déclencher ML en background ───────────────────────────────
    threading.Thread(target=run_pipeline, daemon=True).start()

    return jsonify({"status": "saved", "player_id": data["player_id"]})


# ── API endpoints ─────────────────────────────────────────────────
@app.route('/api/players',          methods=['GET'])
def get_players():
    return jsonify(list(db.players.find({}, {"_id": 0}).sort("timestamp", -1)))

@app.route('/api/ml-results',       methods=['GET'])
def ml_results():
    return jsonify(list(db.players_classified.find({}, {"_id": 0})))

@app.route('/api/fuzzy-results',    methods=['GET'])
def fuzzy_results():
    return jsonify(list(db.fuzzy_results.find({}, {"_id": 0})))

@app.route('/api/metrics',          methods=['GET'])
def metrics():
    return jsonify(db.model_metrics.find_one({}, {"_id": 0}) or {})

@app.route('/api/game-evaluation',  methods=['GET'])
def game_evaluation():
    d = db.game_evaluation.find_one({}, {"_id": 0})
    return jsonify(d) if d else (jsonify({"error": "Pas encore d'évaluation"}), 404)

@app.route('/api/status',           methods=['GET'])
def status():
    return jsonify({
        "players"       : db.players.count_documents({}),
        "classified"    : db.players_classified.count_documents({}),
        "fuzzy_results" : db.fuzzy_results.count_documents({}),
        "has_evaluation": db.game_evaluation.count_documents({}) > 0,
        "timestamp"     : time.time()
    })

@app.route('/')
def home():
    return jsonify({"status": "running", "port": 5000})

if __name__ == '__main__':
    print("🚀 Backend démarré sur http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)