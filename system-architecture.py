from typing import Dict, List, Optional, Tuple
import numpy as np
import redis
from datetime import datetime, timedelta
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Système de Cache Redis
class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        
    async def get_cached_prediction(self, match_id: str) -> Optional[Dict]:
        """Récupère une prédiction cachée si elle existe et est valide"""
        cached_data = self.redis_client.get(f"prediction:{match_id}")
        if cached_data:
            prediction = json.loads(cached_data)
            # Vérifie si la prédiction n'est pas périmée (max 6h)
            if datetime.fromisoformat(prediction['timestamp']) > datetime.now() - timedelta(hours=6):
                return prediction
        return None
        
    async def cache_prediction(self, match_id: str, prediction: Dict):
        """Cache une nouvelle prédiction avec TTL"""
        prediction['timestamp'] = datetime.now().isoformat()
        self.redis_client.setex(
            f"prediction:{match_id}",
            timedelta(hours=6),
            json.dumps(prediction)
        )

# 2. Système de Mise à Jour des Modèles
class ModelUpdateManager:
    def __init__(self):
        self.current_model_version = 1.0
        self.models = {
            'statistical': None,
            'deep_learning': None
        }
        self.update_lock = asyncio.Lock()
        
    async def schedule_model_update(self):
        """Planifie la mise à jour des modèles pendant les heures creuses"""
        while True:
            current_hour = datetime.now().hour
            # Mise à jour à 3h du matin
            if current_hour == 3:
                await self.update_models()
            await asyncio.sleep(3600)  # Vérifie toutes les heures
            
    async def update_models(self):
        """Processus de mise à jour des modèles"""
        async with self.update_lock:
            try:
                logger.info("Début de la mise à jour des modèles")
                
                # 1. Collecte des nouvelles données
                new_data = await self.collect_new_training_data()
                
                # 2. Évaluation des performances actuelles
                current_metrics = await self.evaluate_current_models(new_data)
                
                # 3. Entraînement des nouveaux modèles
                new_models = await self.train_new_models(new_data)
                
                # 4. Évaluation des nouveaux modèles
                new_metrics = await self.evaluate_models(new_models, new_data)
                
                # 5. Décision de mise à jour
                if self.should_update_models(current_metrics, new_metrics):
                    await self.deploy_new_models(new_models)
                    self.current_model_version += 0.1
                    logger.info(f"Modèles mis à jour vers la version {self.current_model_version}")
                
            except Exception as e:
                logger.error(f"Erreur lors de la mise à jour des modèles: {str(e)}")
                
    def should_update_models(self, current_metrics: Dict, new_metrics: Dict) -> bool:
        """Décide si les nouveaux modèles sont meilleurs que les actuels"""
        improvement_threshold = 0.02  # 2% d'amélioration minimum
        
        metrics_comparison = {
            'accuracy': new_metrics['accuracy'] - current_metrics['accuracy'],
            'rmse': current_metrics['rmse'] - new_metrics['rmse'],
            'log_loss': current_metrics['log_loss'] - new_metrics['log_loss']
        }
        
        return all(imp > improvement_threshold for imp in metrics_comparison.values())

# 3. Système de Validation des Prédictions
class PredictionValidator:
    def __init__(self):
        self.validation_thresholds = {
            'confidence_min': 0.6,
            'probability_sum_max': 1.1,
            'max_goals': 10
        }
        
    async def validate_prediction(self, prediction: Dict) -> Tuple[bool, str]:
        """Valide une prédiction selon plusieurs critères"""
        # 1. Vérification de la confiance
        if prediction['confidence'] < self.validation_thresholds['confidence_min']:
            return False, "Confiance insuffisante"
            
        # 2. Vérification de la cohérence des probabilités
        if sum(prediction['probabilities'].values()) > self.validation_thresholds['probability_sum_max']:
            return False, "Probabilités incohérentes"
            
        # 3. Vérification des scores prédits
        if prediction['home_score'] > self.validation_thresholds['max_goals'] or \
           prediction['away_score'] > self.validation_thresholds['max_goals']:
            return False, "Scores prédits anormalement élevés"
            
        # 4. Vérification des données d'entrée
        if not self.validate_input_features(prediction['features']):
            return False, "Données d'entrée invalides ou incomplètes"
            
        return True, "Prédiction valide"
        
    def validate_input_features(self, features: Dict) -> bool:
        """Valide les caractéristiques d'entrée du modèle"""
        required_features = {
            'team_form',
            'historical_performance',
            'player_statistics',
            'weather_conditions'
        }
        
        return all(feature in features for feature in required_features)

# 4. Gestionnaire Principal de Prédictions
class PredictionManager:
    def __init__(self):
        self.cache_manager = CacheManager()
        self.model_updater = ModelUpdateManager()
        self.validator = PredictionValidator()
        
    async def get_prediction(self, match_id: str) -> Dict:
        """Processus complet de prédiction avec cache et validation"""
        # 1. Vérification du cache
        cached_prediction = await self.cache_manager.get_cached_prediction(match_id)
        if cached_prediction:
            return cached_prediction
            
        # 2. Génération de la prédiction
        prediction = await self.generate_prediction(match_id)
        
        # 3. Validation
        is_valid, message = await self.validator.validate_prediction(prediction)
        if not is_valid:
            logger.warning(f"Prédiction invalide pour le match {match_id}: {message}")
            return self.get_fallback_prediction(match_id)
            
        # 4. Mise en cache
        await self.cache_manager.cache_prediction(match_id, prediction)
        
        return prediction
        
    def get_fallback_prediction(self, match_id: str) -> Dict:
        """Retourne une prédiction de repli basée sur des statistiques simples"""
        return {
            'home_score': None,
            'away_score': None,
            'confidence': 0,
            'message': "Prédiction impossible - données insuffisantes"
        }

# 5. API FastAPI
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Démarre la mise à jour programmée des modèles
    asyncio.create_task(app.state.model_updater.schedule_model_update())

@app.get("/api/predictions/{match_id}")
async def get_match_prediction(match_id: str, background_tasks: BackgroundTasks):
    prediction_manager = PredictionManager()
    prediction = await prediction_manager.get_prediction(match_id)
    
    # Planifie la collecte des métriques en arrière-plan
    background_tasks.add_task(collect_prediction_metrics, match_id, prediction)
    
    return prediction
