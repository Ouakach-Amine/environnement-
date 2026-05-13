"""
Backend Flask — Mode temps réel
- Écrit directement dans MongoDB (sans attendre Spark)
- Lance ML + Fuzzy AHP en background immédiatement après chaque joueur
"""
import threading
import time
import subprocess
import sys
import os
import math
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

# ── MongoDB direct ────────────────────────────────────────────────
client = MongoClient("mongodb://localhost:27017/")
db     = client["game_db"]

# ── Chemin vers les scripts ML (même dossier ou ../ml-service) ───
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ML_DIR    = os.path.join(BASE_DIR, "..", "ml-service")
MODEL_PY  = os.path.join(ML_DIR, "model_once.py")
FUZZY_PY  = os.path.join(ML_DIR, "fuzzy_ahp.py")
PYTHON    = sys.executable   # utilise le même Python que le backend

# ── Lock pour éviter deux exécutions ML simultanées ──────────────
_ml_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════
# 🔧 Fonction anti-NaN (AJOUTÉE)
# ═══════════════════════════════════════════════════════════════════
def clean_nan(obj):
    """
    Remplace récursivement NaN, Infinity et -Infinity par None
    pour garantir un JSON valide.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None  # ou 0.0 si vous préférez
        return obj
    elif isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_nan(v) for v in obj)
    else:
        return obj


def run_ml_pipeline():
    """
    Lance model_once.py puis fuzzy_ahp.py en séquence.
    Appelé dans un thread séparé pour ne pas bloquer le backend.
    """
    if not _ml_lock.acquire(blocking=False):
        print("⏳ ML pipeline déjà en cours, skip")
        return
    try:
        print("🔄 ML pipeline démarré...")

        # Attendre qu'il y ait au moins 3 joueurs
        count = db.players.count_documents({})
        if count < 3:
            print(f"⏳ Seulement {count} joueur(s) — ML skip (min 3)")
            return

        # 1. Clustering ML
        r1 = subprocess.run(
            [PYTHON, MODEL_PY],
            capture_output=True, text=True, timeout=60
        )
        if r1.stdout: print("[model_once]", r1.stdout.strip())
        if r1.stderr: print("[model_once ERR]", r1.stderr.strip())

        # 2. Fuzzy AHP
        r2 = subprocess.run(
            [PYTHON, FUZZY_PY],
            capture_output=True, text=True, timeout=60
        )
        if r2.stdout: print("[fuzzy_ahp]", r2.stdout.strip())
        if r2.stderr: print("[fuzzy_ahp ERR]", r2.stderr.strip())

        print("✅ ML pipeline terminé")

    except subprocess.TimeoutExpired:
        print("❌ ML pipeline timeout")
    except Exception as e:
        print(f"❌ ML pipeline erreur: {e}")
    finally:
        _ml_lock.release()


# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route('/send', methods=['POST'])
def send_data():
    req = request.json
    if not req:
        return jsonify({"error": "No JSON body"}), 400

    # Validation des champs obligatoires
    if req.get("player_id") is None or req.get("score") is None:
        return jsonify({"error": "player_id and score are required"}), 400

    data = {
        "player_id"    : req.get("player_id"),
        "score"        : req.get("score"),
        "time"         : req.get("time", 0),
        "clicks"       : req.get("clicks", 0),
        "moves"        : req.get("moves", 0),
        "errors"       : req.get("errors", 0),
        "response_time": req.get("response_time", 0.0),
        "level"        : req.get("level", 1),
        "success"      : req.get("success", 0),
        "repetition"   : req.get("repetition", 0),
        "timestamp"    : time.time()
    }

    # ── Écriture directe dans MongoDB ────────────────────────────
    # Upsert : si le joueur existe déjà, on met à jour
    db.players.update_one(
        {"player_id": data["player_id"]},
        {"$set": data},
        upsert=True
    )
    print(f"✅ Joueur #{data['player_id']} sauvegardé dans MongoDB")

    # ── Lancer ML + Fuzzy AHP en arrière-plan ────────────────────
    t = threading.Thread(target=run_ml_pipeline, daemon=True)
    t.start()

    # Nettoie les NaN avant d'envoyer la réponse (AJOUTÉ)
    return jsonify(clean_nan({"status": "saved", "data": data}))


@app.route('/api/players', methods=['GET'])
def get_players():
    data = list(db.players.find({}, {"_id": 0}).sort("timestamp", -1))
    return jsonify(clean_nan(data))  # MODIFIÉ


@app.route('/api/ml-results', methods=['GET'])
def ml_results():
    data = list(db.players_classified.find({}, {"_id": 0}))
    return jsonify(clean_nan(data))  # MODIFIÉ


@app.route('/api/fuzzy-results', methods=['GET'])
def fuzzy_results():
    data = list(db.fuzzy_results.find({}, {"_id": 0}))
    return jsonify(clean_nan(data))  # MODIFIÉ


@app.route('/api/metrics', methods=['GET'])
def metrics():
    data = db.model_metrics.find_one({}, {"_id": 0})
    return jsonify(clean_nan(data or {}))  # MODIFIÉ


@app.route('/api/game-evaluation', methods=['GET'])
def game_evaluation():
    data = db.game_evaluation.find_one({}, {"_id": 0})
    if not data:
        return jsonify({"error": "No evaluation yet"}), 404
    return jsonify(clean_nan(data))  # MODIFIÉ


@app.route('/api/status', methods=['GET'])
def status():
    """Endpoint de statut : le front le poll pour savoir quand rafraîchir."""
    return jsonify(clean_nan({  # MODIFIÉ
        "players"         : db.players.count_documents({}),
        "classified"      : db.players_classified.count_documents({}),
        "fuzzy_results"   : db.fuzzy_results.count_documents({}),
        "has_evaluation"  : db.game_evaluation.count_documents({}) > 0,
        "timestamp"       : time.time()
    }))


@app.route('/')
def home():
    return jsonify({"status": "running", "port": 5000})


if __name__ == '__main__':
    # Essayer plusieurs ports si le 5000 est occupé
    ports = [5000, 5001, 5002, 5003]
    for port in ports:
        try:
            print(f"🚀 Tentative de démarrage sur le port {port}...")
            app.run(host='0.0.0.0', port=port, debug=False)
            break
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"⚠️ Port {port} occupé, essai du suivant...")
                continue
            raise