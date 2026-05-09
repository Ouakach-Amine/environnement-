"""
fuzzy_main.py
───────────────────────────────────────────────────────────────────────────────
Main runner for the Fuzzy AHP Serious Game Evaluation System.

This script:
  1.  Computes Fuzzy AHP weights once at startup (deterministic).
  2.  Polls MongoDB every POLL_INTERVAL seconds.
  3.  Whenever players_classified contains data, runs the full evaluation
      and writes results back to MongoDB.
"""

import time
import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from fuzzy_evaluation import AHPWeights, run_fuzzy_evaluation

# ── Configuration ─────────────────────────────────────────────────────────────
MONGO_URI      = "mongodb://mongodb:27017/"
DB_NAME        = "game_db"
POLL_INTERVAL  = 15   # seconds between evaluation cycles
MIN_PLAYERS    = 1    # minimum players required to run evaluation


def wait_for_mongo(uri: str, retries: int = 20, delay: float = 5.0):
    """Block until MongoDB is reachable."""
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    for attempt in range(1, retries + 1):
        try:
            client.admin.command("ping")
            print(f"✅  Connected to MongoDB ({uri})")
            return client
        except ConnectionFailure:
            print(f"⏳  MongoDB not ready — attempt {attempt}/{retries}, retrying in {delay}s …")
            time.sleep(delay)
    print("❌  Could not connect to MongoDB. Exiting.")
    sys.exit(1)


def main():
    print("╔══════════════════════════════════════════════╗")
    print("║   Fuzzy AHP Serious Game Evaluation System   ║")
    print("╚══════════════════════════════════════════════╝\n")

    # 1. Connect to MongoDB
    client = wait_for_mongo(MONGO_URI)
    db     = client[DB_NAME]

    # 2. Pre-compute Fuzzy AHP weights (fixed expert matrices — run once)
    weights = AHPWeights()

    # 3. Main evaluation loop
    cycle = 0
    while True:
        cycle += 1
        print(f"\n── Evaluation cycle #{cycle} ({'─' * 30})")

        try:
            n_players = db.players_classified.count_documents({})

            if n_players < MIN_PLAYERS:
                print(f"  ⏳  Only {n_players} classified player(s) — need ≥ {MIN_PLAYERS}")
            else:
                print(f"  👥  Found {n_players} classified player(s) — running evaluation …")
                run_fuzzy_evaluation(db, weights)

        except Exception as exc:
            print(f"  ❌  Error in evaluation cycle: {exc}")

        print(f"  💤  Sleeping {POLL_INTERVAL}s …")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
