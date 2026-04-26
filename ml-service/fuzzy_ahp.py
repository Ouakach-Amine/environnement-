from pymongo import MongoClient
import numpy as np

client = MongoClient("mongodb://mongodb:27017/")
db = client["game_db"]

data = list(db.players_classified.find({}, {"_id": 0}))

if len(data) == 0:
    print("❌ No classified data")
    exit()

# poids AHP
weights = {
    "pedagogique": 0.5,
    "technique": 0.25,
    "ludique": 0.15,
    "comportemental": 0.10
}

# fonctions floues simples
def fuzzy_score(score):
    if score < 40:
        return 0.2
    elif score < 70:
        return 0.6
    else:
        return 1.0

def fuzzy_errors(errors):
    if errors > 5:
        return 0.2
    elif errors > 2:
        return 0.6
    else:
        return 1.0

def fuzzy_time(t):
    if t > 30:
        return 0.3
    else:
        return 1.0

results = []

raw_data = list(db.players.find({}, {"_id": 0}))

for player in raw_data:
    ped = fuzzy_score(player.get("score", 0))
    tech = fuzzy_time(player.get("time", 0))
    lud = min(player.get("clicks", 0) / 50, 1)
    comp = fuzzy_errors(player.get("errors", 0))

    # score global
    final_score = (
        ped * weights["pedagogique"] +
        tech * weights["technique"] +
        lud * weights["ludique"] +
        comp * weights["comportemental"]
    )

    # décision
    if final_score > 0.75:
        decision = "expert"
    elif final_score > 0.5:
        decision = "intermediate"
    else:
        decision = "beginner"

    results.append({
        "player_id": player["player_id"],
        "fuzzy_score": final_score,
        "decision": decision
    })

db.fuzzy_results.delete_many({})
db.fuzzy_results.insert_many(results)

print("✅ Fuzzy AHP results saved")
