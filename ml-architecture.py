from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
import torch.nn as nn

@dataclass
class MatchFeatures:
    home_team_form: List[float]
    away_team_form: List[float]
    head_to_head: List[Dict]
    home_team_stats: Dict
    away_team_stats: Dict
    context_features: Dict
    weather_conditions: Optional[Dict]

class GoalsPredictionModel:
    def __init__(self):
        self.poisson_model = PoissonRegressor()
        self.over_under_model = RandomForestClassifier()
        
    def preprocess_features(self, match_features: MatchFeatures) -> np.ndarray:
        # Convertit les caractéristiques brutes en variables numériques
        processed_features = []
        # Forme récente (moyenne des 5 derniers matchs)
        processed_features.extend([
            np.mean(match_features.home_team_form),
            np.mean(match_features.away_team_form)
        ])
        # Statistiques d'équipe
        processed_features.extend([
            match_features.home_team_stats['goals_scored_avg'],
            match_features.home_team_stats['goals_conceded_avg'],
            match_features.away_team_stats['goals_scored_avg'],
            match_features.away_team_stats['goals_conceded_avg']
        ])
        return np.array(processed_features)

    def predict_score(self, features: np.ndarray) -> tuple:
        # Prédit le nombre de buts pour chaque équipe
        home_goals = self.poisson_model.predict(features)
        away_goals = self.poisson_model.predict(features)
        return home_goals, away_goals

    def predict_over_under(self, features: np.ndarray) -> float:
        # Prédit la probabilité de plus de 1.5 buts
        return self.over_under_model.predict_proba(features)[:, 1]

class DeepLearningModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # [home_goals, away_goals, over_15_prob]
        )
    
    def forward(self, x):
        return self.network(x)

class PredictionService:
    def __init__(self):
        self.statistical_model = GoalsPredictionModel()
        self.deep_model = DeepLearningModel(input_size=32)
        
    async def get_prediction(self, match_id: str) -> Dict:
        # Récupère les données du match
        match_features = await self.fetch_match_features(match_id)
        
        # Prétraitement
        processed_features = self.statistical_model.preprocess_features(match_features)
        
        # Combine les prédictions des deux modèles
        stat_home, stat_away = self.statistical_model.predict_score(processed_features)
        stat_over = self.statistical_model.predict_over_under(processed_features)
        
        dl_predictions = self.deep_model(torch.tensor(processed_features))
        dl_home, dl_away, dl_over = dl_predictions.detach().numpy()
        
        # Moyenne pondérée des prédictions
        return {
            'home_score': (stat_home * 0.6 + dl_home * 0.4),
            'away_score': (stat_away * 0.6 + dl_away * 0.4),
            'over_15_probability': (stat_over * 0.6 + dl_over * 0.4)
        }
