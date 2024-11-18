from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from prometheus_client import Counter, Gauge, Histogram, Summary
import elasticapm
from loguru import logger

@dataclass
class AlertThreshold:
    warning: float
    critical: float
    duration: str  # Durée pendant laquelle le seuil doit être dépassé
    cooldown: str  # Période minimale entre deux alertes

# 1. Configuration des seuils d'alerte
class AlertConfig:
    def __init__(self):
        self.thresholds = {
            # Performances système
            'cpu_usage': AlertThreshold(
                warning=70.0,
                critical=85.0,
                duration='5m',
                cooldown='15m'
            ),
            'memory_usage': AlertThreshold(
                warning=75.0,
                critical=90.0,
                duration='5m',
                cooldown='15m'
            ),
            'disk_usage': AlertThreshold(
                warning=80.0,
                critical=90.0,
                duration='5m',
                cooldown='1h'
            ),
            
            # Performances applicatives
            'prediction_latency': AlertThreshold(
                warning=1.5,  # secondes
                critical=3.0,
                duration='5m',
                cooldown='15m'
            ),
            'error_rate': AlertThreshold(
                warning=5.0,  # pourcentage
                critical=10.0,
                duration='5m',
                cooldown='15m'
            ),
            
            # Performances ML
            'model_accuracy': AlertThreshold(
                warning=0.65,
                critical=0.60,
                duration='1h',
                cooldown='6h'
            ),
            'prediction_deviation': AlertThreshold(
                warning=15.0,  # pourcentage
                critical=25.0,
                duration='1h',
                cooldown='6h'
            ),
            
            # Infrastructure
            'database_connections': AlertThreshold(
                warning=80.0,
                critical=90.0,
                duration='5m',
                cooldown='15m'
            ),
            'cache_hit_rate': AlertThreshold(
                warning=70.0,  # pourcentage
                critical=50.0,
                duration='15m',
                cooldown='1h'
            )
        }

# 2. Métriques détaillées
class MetricsCollector:
    def __init__(self):
        # Métriques système
        self.system_metrics = {
            'cpu_usage': Gauge('system_cpu_usage', 'CPU usage in percentage'),
            'memory_usage': Gauge('system_memory_usage', 'Memory usage in percentage'),
            'disk_usage': Gauge('system_disk_usage', 'Disk usage in percentage'),
        }
        
        # Métriques applicatives
        self.app_metrics = {
            'requests_total': Counter('app_requests_total', 'Total HTTP requests'),
            'request_duration': Histogram(
                'app_request_duration_seconds',
                'Request duration in seconds',
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
            ),
            'active_users': Gauge('app_active_users', 'Number of active users'),
        }
        
        # Métriques ML
        self.ml_metrics = {
            'prediction_accuracy': Gauge('ml_prediction_accuracy', 'Model prediction accuracy'),
            'training_duration': Summary('ml_training_duration_seconds', 'Model training duration'),
            'feature_importance': Gauge('ml_feature_importance', 'Feature importance scores'),
        }
        
        # Métriques business
        self.business_metrics = {
            'predictions_count': Counter('business_predictions_total', 'Total predictions made'),
            'successful_bets': Counter('business_successful_bets', 'Number of successful predictions'),
            'conversion_rate': Gauge('business_conversion_rate', 'User conversion rate'),
        }

# 3. Système de Disaster Recovery
class DisasterRecoveryManager:
    def __init__(self):
        self.apm_client = elasticapm.Client()
        
    async def handle_disaster(self, incident_type: str):
        """Gère les scénarios de disaster recovery"""
        try:
            # 1. Détection et classification de l'incident
            severity = self._classify_incident(incident_type)
            
            # 2. Activation du plan de recovery approprié
            recovery_plan = self._get_recovery_plan(incident_type, severity)
            
            # 3. Exécution du plan
            for step in recovery_plan:
                success = await self._execute_recovery_step(step)
                if not success:
                    await self._escalate_incident(step, incident_type)
                
            # 4. Vérification post-recovery
            if await self._verify_system_health():
                await self._send_recovery_report(incident_type)
            else:
                await self._initiate_manual_intervention()
                
        except Exception as e:
            await self._handle_recovery_failure(e)
    
    async def _execute_recovery_step(self, step: Dict):
        """Exécute une étape du plan de recovery"""
        logger.info(f"Exécution de l'étape de recovery: {step['name']}")
        
        try:
            if step['type'] == 'failover':
                await self._execute_failover(step)
            elif step['type'] == 'restore':
                await self._execute_restore(step)
            elif step['type'] == 'reconfig':
                await self._execute_reconfig(step)
            
            return await self._verify_step_success(step)
            
        except Exception as e:
            logger.error(f"Échec de l'étape {step['name']}: {str(e)}")
            return False

# 4. Intégration CI/CD
class CICDIntegration:
    def __init__(self):
        self.github_actions = GithubActionsClient()
        self.jenkins = JenkinsClient()
        self.argocd = ArgoCDClient()
        
    async def deploy_pipeline(self, version: str):
        """Pipeline de déploiement complet"""
        try:
            # 1. Tests automatisés
            test_results = await self._run_test_suite()
            if not test_results['success']:
                raise DeploymentError(f"Tests failed: {test_results['failures']}")
            
            # 2. Build et push des images Docker
            image_tags = await self._build_and_push_images(version)
            
            # 3. Mise à jour des manifests Kubernetes
            await self._update_k8s_manifests(version, image_tags)
            
            # 4. Déploiement progressif
            await self._rolling_deployment(version)
            
            # 5. Tests de smoke post-déploiement
            if not await self._smoke_tests():
                await self._trigger_rollback(version)
                
        except Exception as e:
            await self._handle_deployment_failure(e)
            
    async def _smoke_tests(self) -> bool:
        """Tests de base post-déploiement"""
        tests = [
            self._test_api_endpoints(),
            self._test_ml_predictions(),
            self._test_database_connectivity(),
            self._test_cache_operation()
        ]
        
        results = await asyncio.gather(*tests)
        return all(results)

# 5. Système de Logging
class LoggingSystem:
    def __init__(self):
        self.elk_client = ElasticsearchClient()
        self.logger = self._configure_logger()
        
    def _configure_logger(self):
        """Configuration du système de logging"""
        config = {
            "handlers": [
                {"sink": "logs/app.log", "rotation": "500 MB"},
                {"sink": "logs/error.log", "level": "ERROR"},
                {
                    "sink": self.elk_client.send,
                    "serialize": self._serialize_log,
                    "level": "INFO"
                }
            ],
            "extra": {"app_name": "football-predictions"}
        }
        
        return logger.configure(**config)
        
    async def log_event(self, event_type: str, data: Dict):
        """Log un événement avec contexte"""
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "environment": os.getenv("ENVIRONMENT", "production"),
            "version": os.getenv("APP_VERSION"),
            "data": data
        }
        
        self.logger.bind(**context).info(f"Event logged: {event_type}")
