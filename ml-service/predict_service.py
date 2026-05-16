"""
predict_service.py - Service de prédiction avec les 31 features du modèle
"""
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Les 31 features exactes utilisées pour l'entraînement
MODEL_FEATURES = [
    # Pedagogical
    "hints_used",
    "correct_answers",
    "wrong_answers",
    "objectives_completed",
    "knowledge_score",
    "progression_rate",
    "retry_after_fail",

    # Technological
    "load_time",
    "crash_count",
    "lag_events",
    "frame_drops",
    "api_errors",
    "device_type",
    "screen_width",
    "screen_height",

    # Ludic
    "playtime_voluntary",
    "bonus_collected",
    "challenges_attempted",
    "idle_time",
    "exploration_rate",
    "combo_count",
    "skip_count",

    # Behavioural
    "help_requests",
    "give_up_count",
    "pause_count",
    "total_pause_time",
    "focus_time",
    "frustration_events",
    "session_count",
    "days_active"
]

class PlayerPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "player_classification_kmeans.pkl"
        
        self.model_path = str(model_path)
        self.model = None
        self.scaler = None
        self.load_model()
        
    def load_model(self):
        """Charge le modèle KMeans sauvegardé"""
        try:
            data = joblib.load(self.model_path)
            print(f"✅ Modèle chargé avec joblib")
        except:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Modèle chargé avec pickle")
        
        print(f"   Type: {type(data).__name__}")
        
        if isinstance(data, dict):
            print(f"   Clés disponibles: {list(data.keys())}")
            
            # Chercher le modèle KMeans
            for key in ['model', 'kmeans', 'classifier', 'estimator']:
                if key in data and hasattr(data[key], 'predict'):
                    self.model = data[key]
                    print(f"   ✅ Modèle trouvé: {key}")
                    break
            
            # Chercher le scaler
            for key in ['scaler', 'standard_scaler', 'preprocessor']:
                if key in data:
                    self.scaler = data[key]
                    print(f"   ✅ Scaler trouvé: {key}")
                    break
        else:
            self.model = data
        
        if self.model is None:
            raise ValueError("❌ Aucun modèle trouvé dans le fichier")
        
        print(f"   ✅ Modèle prêt: {type(self.model).__name__}")
    
    def predict(self, player_data):
        """
        Prédit le cluster/niveau d'un joueur
        
        Args:
            player_data: dict avec les données du joueur
        
        Returns:
            dict avec cluster, niveau et type de modèle
        """
        # Créer un DataFrame avec les features dans le bon ordre
        features_dict = {}
        for feature in MODEL_FEATURES:
            features_dict[feature] = player_data.get(feature, 0)
        
        df = pd.DataFrame([features_dict])
        
        # Appliquer le scaler si disponible
        if self.scaler is not None:
            X = self.scaler.transform(df)
        else:
            # Sinon normaliser manuellement
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(df)
        
        # Prédire le cluster
        cluster = int(self.model.predict(X)[0])
        
        # Calculer la distance aux centroids pour la confiance
        confidence = None
        if hasattr(self.model, 'transform'):
            distances = self.model.transform(X)[0]
            confidence = float(1 / (1 + distances[cluster]))
        
        # Mapping cluster -> niveau
        # À ajuster selon votre logique métier
        level_map = {
            0: 'beginner',
            1: 'intermediate', 
            2: 'expert'
        }
        level = level_map.get(cluster, 'intermediate')
        
        return {
            'cluster': cluster,
            'level': level,
            'confidence': confidence,
            'model_type': type(self.model).__name__
        }

# Instance globale
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = PlayerPredictor()
    return _predictor

def predict_player(player_data):
    """Fonction rapide pour prédire"""
    predictor = get_predictor()
    return predictor.predict(player_data)

# Test
if __name__ == "__main__":
    # Simuler un joueur avec toutes les features
    test_player = {
        # Pedagogical
        "hints_used": 3,
        "correct_answers": 15,
        "wrong_answers": 5,
        "objectives_completed": 8,
        "knowledge_score": 75.5,
        "progression_rate": 0.65,
        "retry_after_fail": 2,
        
        # Technological
        "load_time": 1.2,
        "crash_count": 0,
        "lag_events": 1,
        "frame_drops": 3,
        "api_errors": 0,
        "device_type": 0,
        "screen_width": 1920,
        "screen_height": 1080,
        
        # Ludic
        "playtime_voluntary": 45.5,
        "bonus_collected": 5,
        "challenges_attempted": 3,
        "idle_time": 2.5,
        "exploration_rate": 0.7,
        "combo_count": 4,
        "skip_count": 1,
        
        # Behavioural
        "help_requests": 2,
        "give_up_count": 0,
        "pause_count": 1,
        "total_pause_time": 15.0,
        "focus_time": 120.0,
        "frustration_events": 1,
        "session_count": 3,
        "days_active": 2
    }
    
    print("\n🧪 Test de prédiction avec les 31 features:")
    result = predict_player(test_player)
    print(f"   Cluster: {result['cluster']}")
    print(f"   Niveau: {result['level']}")
    print(f"   Confiance: {result['confidence']}")