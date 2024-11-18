import docker
import prometheus_client as prom
import alertmanager_client as alert
from kubernetes import client, config
from pathlib import Path
import shutil
import boto3
from datetime import datetime
import schedule
import time

# 1. Système de Backup et Récupération
class BackupManager:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.backup_bucket = 'football-predictions-backups'
        self.local_backup_path = Path('/var/backups/football-predictions')
        
    async def create_full_backup(self):
        """Crée une sauvegarde complète du système"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # 1. Backup des données PostgreSQL
            await self._backup_database(timestamp)
            
            # 2. Backup des modèles ML
            await self._backup_ml_models(timestamp)
            
            # 3. Backup de la configuration
            await self._backup_config(timestamp)
            
            # 4. Upload vers S3
            self._upload_to_s3(timestamp)
            
            # 5. Rotation des backups (garde les 7 derniers jours)
            self._rotate_backups()
            
            logger.info(f"Backup complet créé avec succès: {timestamp}")
            
        except Exception as e:
            logger.error(f"Erreur lors du backup: {str(e)}")
            alert_manager.send_alert(
                severity="critical",
                message=f"Échec du backup: {str(e)}"
            )
    
    async def restore_from_backup(self, timestamp: str):
        """Restaure le système depuis une sauvegarde"""
        try:
            # 1. Téléchargement depuis S3
            backup_files = self._download_from_s3(timestamp)
            
            # 2. Validation du backup
            if not self._validate_backup(backup_files):
                raise ValueError("Backup corrompu ou incomplet")
            
            # 3. Arrêt des services
            await self._stop_services()
            
            # 4. Restauration des données
            await self._restore_database(backup_files['db'])
            await self._restore_ml_models(backup_files['models'])
            await self._restore_config(backup_files['config'])
            
            # 5. Redémarrage des services
            await self._start_services()
            
            logger.info(f"Restauration réussie depuis {timestamp}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la restauration: {str(e)}")
            await self._emergency_recovery()

# 2. Système de Monitoring et Alerting
class MonitoringSystem:
    def __init__(self):
        # Métriques Prometheus
        self.prediction_latency = prom.Histogram(
            'prediction_latency_seconds',
            'Temps de génération des prédictions',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
        )
        
        self.prediction_accuracy = prom.Gauge(
            'prediction_accuracy',
            'Précision des prédictions sur les dernières 24h'
        )
        
        self.model_version = prom.Gauge(
            'model_version',
            'Version actuelle des modèles ML'
        )
        
        # Alerting
        self.alert_manager = alert.AlertManager(
            host='alertmanager:9093',
            environment='production'
        )
        
    async def monitor_system_health(self):
        """Surveillance continue du système"""
        while True:
            # 1. Vérification des ressources système
            system_metrics = await self._check_system_resources()
            if system_metrics['cpu_usage'] > 80:
                self.alert_manager.send_alert(
                    severity="warning",
                    message="Usage CPU élevé",
                    metrics=system_metrics
                )
            
            # 2. Vérification de la latence des prédictions
            latency_metrics = await self._check_prediction_latency()
            if latency_metrics['p95'] > 2.0:
                self.alert_manager.send_alert(
                    severity="warning",
                    message="Latence élevée des prédictions"
                )
            
            # 3. Vérification de la précision des modèles
            accuracy_metrics = await self._check_model_accuracy()
            if accuracy_metrics['accuracy_24h'] < 0.6:
                self.alert_manager.send_alert(
                    severity="critical",
                    message="Baisse significative de la précision"
                )
            
            await asyncio.sleep(60)  # Vérifification toutes les minutes
    
    async def generate_daily_report(self):
        """Génère un rapport quotidien des performances"""
        metrics = {
            'system': await self._collect_system_metrics(),
            'predictions': await self._collect_prediction_metrics(),
            'models': await self._collect_model_metrics()
        }
        
        report = self._format_daily_report(metrics)
        await self._send_report(report)

# 3. Système de Déploiement et Rollback
class DeploymentManager:
    def __init__(self):
        self.docker_client = docker.from_env()
        config.load_kube_config()
        self.k8s_client = client.CoreV1Api()
        
    async def deploy_new_version(self, version: str):
        """Déploie une nouvelle version du système"""
        try:
            # 1. Validation pré-déploiement
            if not await self._validate_deployment(version):
                raise ValueError("Validation pré-déploiement échouée")
            
            # 2. Backup de sécurité
            await backup_manager.create_full_backup()
            
            # 3. Déploiement progressif (Blue/Green)
            deployment_config = self._prepare_deployment(version)
            
            # 3.1 Déploiement de la nouvelle version (Green)
            await self._deploy_green_environment(deployment_config)
            
            # 3.2 Tests de la nouvelle version
            if await self._test_deployment():
                # 3.3 Bascule du trafic
                await self._switch_traffic(version)
                # 3.4 Suppression ancienne version
                await self._cleanup_blue_environment()
            else:
                raise Exception("Tests de déploiement échoués")
            
        except Exception as e:
            logger.error(f"Erreur lors du déploiement: {str(e)}")
            await self.rollback_deployment()
    
    async def rollback_deployment(self):
        """Retour à la version précédente en cas de problème"""
        try:
            # 1. Identification de la dernière version stable
            last_stable = await self._get_last_stable_version()
            
            # 2. Restauration de la configuration
            await self._restore_configuration(last_stable)
            
            # 3. Redémarrage des services critiques
            await self._restart_critical_services()
            
            # 4. Vérification post-rollback
            if await self._verify_system_health():
                logger.info(f"Rollback réussi vers {last_stable}")
            else:
                raise Exception("Échec de la vérification post-rollback")
                
        except Exception as e:
            logger.critical(f"Échec critique du rollback: {str(e)}")
            # Notification d'urgence à l'équipe DevOps
            self.alert_manager.send_emergency_alert()

# 4. Planification des tâches de maintenance
def schedule_maintenance_tasks():
    """Configure les tâches de maintenance régulières"""
    
    # Backups quotidiens à 2h du matin
    schedule.every().day.at("02:00").do(
        backup_manager.create_full_backup
    )
    
    # Rapport quotidien à 6h du matin
    schedule.every().day.at("06:00").do(
        monitoring_system.generate_daily_report
    )
    
    # Vérification hebdomadaire des vieux backups
    schedule.every().monday.do(
        backup_manager._cleanup_old_backups
    )
    
    while True:
        schedule.run_pending()
        time.sleep(60)
