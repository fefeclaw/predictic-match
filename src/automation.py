"""
Automation Module pour Predictic Match

Orchestration complète du pipeline de prédiction footballistique :
- Téléchargement automatique des données
- Feature engineering
- Entraînement/prédiction modèle
- Intégration Polymarket
- Notifications et scheduling

Usage:
    python -m src.automation --league E0 --season 2024 --output output/predictions
"""

import os
import sys
import argparse
import logging
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import des composants du pipeline
from .data_loader import FootballDataLoader
from .data_cleaner import DataCleaner
from .feature_engineer import FeatureEngineer
from .model import prepare_model_data, train_and_evaluate, build_ensemble, save_model, load_model
from .polymarket_client import PolymarketClient, FootballMarket

# Configuration du logging
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES ET CONFIGURATION
# =============================================================================

# URLs football-data.co.uk
FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/"

# Mapping des ligues pour les URLs
LEAGUE_URL_MAPPING = {
    'E0': 'premierleague',
    'E1': 'championship',
    'E2': 'league1',
    'E3': 'league2',
    'SP1': 'laliga',
    'SP2': 'segundaliga',
    'I1': 'seriea',
    'I2': 'serieb',
    'D1': 'bundesliga',
    'D2': 'bundesliga2',
    'F1': 'ligue1',
    'F2': 'ligue2',
    'N1': 'eredivisie',
    'P1': 'primeiraliga',
}


class PipelineStatus(Enum):
    """Statuts possibles du pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class PipelineResult:
    """Résultat d'exécution du pipeline."""
    status: PipelineStatus
    timestamp: str
    league: str
    season: int
    matches_processed: int
    predictions_generated: int
    polymarket_markets: int
    model_accuracy: Optional[float]
    output_path: str
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        result = asdict(self)
        result['status'] = self.status.value
        return result
    
    def to_json(self) -> str:
        """Sérialise en JSON."""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# CLASSE PredictionPipeline
# =============================================================================

class PredictionPipeline:
    """
    Orchestre le pipeline complet de prédiction footballistique.
    
    Étapes :
    1. Chargement des données (football-data.co.uk)
    2. Nettoyage et validation
    3. Feature engineering
    4. Entraînement/chargement du modèle
    5. Génération des prédictions
    6. Récupération données Polymarket
    7. Sauvegarde des résultats
    
    Attributes:
        league: Code de la ligue (E0, SP1, etc.)
        season: Saison (année de début, ex: 2024 pour 2024-2025)
        output_dir: Répertoire de sortie
        data_dir: Répertoire des données
    """
    
    def __init__(
        self,
        league: str = 'E0',
        season: Optional[int] = None,
        output_dir: str = 'output/predictions',
        data_dir: str = 'data/raw',
        models_dir: str = 'data/models',
        use_existing_model: bool = True
    ):
        """
        Initialise le pipeline.
        
        Args:
            league: Code de la ligue (défaut: 'E0' - Premier League)
            season: Saison (défaut: saison actuelle)
            output_dir: Répertoire pour les prédictions
            data_dir: Répertoire des données brutes
            models_dir: Répertoire des modèles entraînés
            use_existing_model: Utiliser modèle existant si disponible
        """
        self.league = league
        self.season = season or self._get_current_season()
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.use_existing_model = use_existing_model
        
        # Création des répertoires
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialisation des composants
        self.loader = FootballDataLoader(
            raw_data_path=str(self.data_dir),
            leagues=[league],
            seasons=list(range(2018, self.season + 1))
        )
        self.cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer(rolling_windows=[5, 10, 20])
        self.polymarket_client = PolymarketClient()
        
        # État du pipeline
        self.current_data: Optional[pd.DataFrame] = None
        self.model = None
        self.model_info: Dict[str, Any] = {}
        
        logger.info(
            f"PredictionPipeline initialized - "
            f"League: {league}, Season: {self.season}"
        )
    
    def _get_current_season(self) -> int:
        """Détermine la saison actuelle."""
        now = datetime.now()
        # Saison commence en août, donc si mois < 8, saison = année - 1
        if now.month < 8:
            return now.year - 1
        return now.year
    
    def _get_model_path(self) -> Path:
        """Retourne le chemin du modèle pour cette ligue/saison."""
        return self.models_dir / f"model_{self.league}_{self.season}.pkl"
    
    def run(self, retrain: bool = False) -> PipelineResult:
        """
        Exécute le pipeline complet.
        
        Args:
            retrain: Force le réentraînement du modèle
            
        Returns:
            PipelineResult avec les statistiques d'exécution
        """
        start_time = datetime.now()
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            timestamp=start_time.isoformat(),
            league=self.league,
            season=self.season,
            matches_processed=0,
            predictions_generated=0,
            polymarket_markets=0,
            model_accuracy=None,
            output_path=str(self.output_dir)
        )
        
        try:
            # Étape 1: Chargement des données
            logger.info("Étape 1/6: Chargement des données...")
            self.current_data = self._load_data()
            result.matches_processed = len(self.current_data)
            
            # Étape 2: Nettoyage
            logger.info("Étape 2/6: Nettoyage des données...")
            self.current_data = self._clean_data()
            
            # Étape 3: Feature engineering
            logger.info("Étape 3/6: Feature engineering...")
            self.current_data = self._create_features()
            
            # Étape 4: Modèle (entraînement ou chargement)
            logger.info("Étape 4/6: Préparation du modèle...")
            model_accuracy = self._prepare_model(retrain=retrain)
            result.model_accuracy = model_accuracy
            
            # Étape 5: Prédictions
            logger.info("Étape 5/6: Génération des prédictions...")
            predictions_count = self._generate_predictions()
            result.predictions_generated = predictions_count
            
            # Étape 6: Polymarket
            logger.info("Étape 6/6: Récupération données Polymarket...")
            markets_count = self._fetch_polymarket_data()
            result.polymarket_markets = markets_count
            
            # Succès
            result.status = PipelineStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            result.status = PipelineStatus.FAILED
            result.error_message = str(e)
        
        # Sauvegarde du rapport
        self._save_pipeline_report(result)
        
        elapsed = datetime.now() - start_time
        logger.info(f"Pipeline completed in {elapsed} - Status: {result.status.value}")
        
        return result
    
    def _load_data(self) -> pd.DataFrame:
        """Charge les données depuis football-data.co.uk."""
        # Tente de charger les données locales
        df = self.loader.load_all_data(ignore_missing=True)
        
        if len(df) == 0:
            logger.warning("Aucune donnée locale trouvée, tentative de téléchargement...")
            fetch_latest_data(self.league, self.season, str(self.data_dir))
            df = self.loader.load_all_data(ignore_missing=True)
        
        if len(df) == 0:
            raise ValueError("Aucune donnée disponible pour cette ligue/saison")
        
        logger.info(f"Données chargées: {len(df)} matchs")
        return df
    
    def _clean_data(self) -> pd.DataFrame:
        """Nettoie et valide les données."""
        df = self.cleaner.clean_data(self.current_data)
        df = self.cleaner.validate_data(df)
        
        logger.info(f"Données nettoyées: {len(df)} matchs valides")
        return df
    
    def _create_features(self) -> pd.DataFrame:
        """Crée les features pour le modèle."""
        df = self.feature_engineer.create_team_features(self.current_data)
        self.feature_columns = self.feature_engineer.get_feature_columns()
        
        # Supprime les lignes avec NaN (début de saison)
        df = df.dropna(subset=self.feature_columns)
        
        logger.info(f"Features créées: {len(self.feature_columns)} colonnes, {len(df)} matchs")
        return df
    
    def _prepare_model(self, retrain: bool = False) -> Optional[float]:
        """Prépare ou entraîne le modèle."""
        model_path = self._get_model_path()
        
        # Vérifie modèle existant
        if model_path.exists() and self.use_existing_model and not retrain:
            logger.info(f"Chargement du modèle existant: {model_path}")
            try:
                self.model, model_metadata = load_model(str(model_path))
                self.model_info = {'source': 'existing', 'path': str(model_path), **model_metadata}
                return model_metadata.get('accuracy')
            except Exception as e:
                logger.warning(f"Échec chargement modèle: {e}. Réentraînement...")
        
        # Entraînement
        logger.info("Entraînement du modèle...")
        
        # Préparation des données
        data = prepare_model_data(
            self.current_data,
            self.feature_columns,
            target_col='FTR_encoded',
            test_size=0.2
        )
        
        # Entraînement et évaluation
        results = train_and_evaluate(data)
        
        # Sélection du meilleur modèle
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_accuracy_mean'])
        best_accuracy = results[best_model_name]['cv_accuracy_mean']
        
        logger.info(f"Meilleur modèle individuel: {best_model_name} (accuracy: {best_accuracy:.3f})")
        
        # Construction d'un ensemble pour meilleure performance
        try:
            ensemble_result = build_ensemble(data, results, voting='soft', use_weights=True)
            self.model = ensemble_result['model']
            ensemble_accuracy = ensemble_result['cv_accuracy_mean']
            logger.info(f"Ensemble créé (accuracy: {ensemble_accuracy:.3f})")
            final_accuracy = ensemble_accuracy
        except Exception as e:
            logger.warning(f"Échec ensemble, utilisation modèle individuel: {e}")
            self.model = results[best_model_name]['model']
            final_accuracy = best_accuracy
        
        # Sauvegarde
        metadata = {
            'name': best_model_name,
            'accuracy': float(final_accuracy),
            'league': self.league,
            'season': self.season,
            'feature_columns': self.feature_columns,
            'trained_at': datetime.now().isoformat()
        }
        save_model(self.model, str(model_path), metadata)
        
        self.model_info = {
            'source': 'trained',
            'name': best_model_name,
            'accuracy': final_accuracy,
            'path': str(model_path),
            'scaler': data.get('scaler')
        }
        
        return float(final_accuracy)
    
    def _generate_predictions(self) -> int:
        """Génère les prédictions pour les prochains matchs."""
        import joblib
        
        # Filtrer les matchs futurs (si disponibles)
        # Pour l'instant, on prédit sur les derniers matchs du dataset
        recent_matches = self.current_data.tail(10).copy()
        
        if len(recent_matches) == 0:
            logger.warning("Aucun match récent pour les prédictions")
            return 0
        
        # Préparation
        X = recent_matches[self.feature_columns].values
        X_scaled = self.model_info.get('scaler', None)
        
        # Prédiction
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)
            recent_matches['pred_home'] = probs[:, 0] if probs.shape[1] > 0 else 0
            recent_matches['pred_draw'] = probs[:, 1] if probs.shape[1] > 1 else 0
            recent_matches['pred_away'] = probs[:, 2] if probs.shape[1] > 2 else 0
        
        recent_matches['pred_result'] = self.model.predict(X)
        
        # Mapping des prédictions
        pred_mapping = {0: 'A', 1: 'D', 2: 'H'}
        recent_matches['pred_FTR'] = recent_matches['pred_result'].map(pred_mapping)
        
        # Sauvegarde
        output_file = self.output_dir / f"predictions_{self.league}_{self.season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        recent_matches.to_csv(output_file, index=False)
        
        logger.info(f"Prédictions sauvegardées: {output_file}")
        return len(recent_matches)
    
    def _fetch_polymarket_data(self) -> int:
        """Récupère les données Polymarket pour la ligue."""
        try:
            # Recherche des marchés pour la ligue
            league_name = FootballDataLoader.LEAGUE_MAPPING.get(self.league, self.league)
            markets = self.polymarket_client.search_markets(
                query=league_name,
                category="Sports"
            )
            
            if markets:
                # Sauvegarde
                output_file = self.output_dir / f"polymarket_{self.league}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                markets_data = []
                for market in markets:
                    markets_data.append({
                        'event_id': market.event_id,
                        'title': market.title,
                        'outcomes': [
                            {'name': o.name, 'price': o.price, 'volume': o.volume}
                            for o in market.outcomes
                        ],
                        'volume': market.volume,
                        'url': market.url
                    })
                
                with open(output_file, 'w') as f:
                    json.dump(markets_data, f, indent=2)
                
                logger.info(f"Polymarket: {len(markets)} marchés sauvegardés")
                return len(markets)
            
        except Exception as e:
            logger.warning(f"Erreur Polymarket: {str(e)}")
        
        return 0
    
    def _save_pipeline_report(self, result: PipelineResult):
        """Sauvegarde le rapport d'exécution."""
        report_file = self.output_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            f.write(result.to_json())
        
        logger.info(f"Rapport sauvegardé: {report_file}")


# =============================================================================
# FONCTIONS DE TÉLÉCHARGEMENT
# =============================================================================

def fetch_latest_data(
    league: str,
    season: int,
    output_dir: str = 'data/raw',
    force: bool = False
) -> Optional[str]:
    """
    Télécharge les dernières données depuis football-data.co.uk.
    
    Args:
        league: Code de la ligue (E0, SP1, etc.)
        season: Saison (ex: 2024)
        output_dir: Répertoire de destination
        force: Force le téléchargement même si fichier existe
        
    Returns:
        Chemin du fichier téléchargé ou None en cas d'échec
    """
    # Vérifie code ligue
    if league not in LEAGUE_URL_MAPPING:
        logger.error(f"Code ligue inconnu: {league}")
        return None
    
    # Construction URL
    season_suffix = str(season)[-2:]
    filename = f"{league.lower()}{season_suffix}.csv"
    url = f"{FOOTBALL_DATA_BASE_URL}{LEAGUE_URL_MAPPING[league]}.php"
    
    # Paramètres de téléchargement
    # football-data.co.uk utilise un système de téléchargement par formulaire
    # On utilise l'URL directe si disponible
    direct_url = f"{FOOTBALL_DATA_BASE_URL}mmz4281/{season_suffix}{season_suffix}/{filename}"
    
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Vérifie fichier existant
    if output_path.exists() and not force:
        logger.info(f"Fichier déjà présent: {output_path}")
        return str(output_path)
    
    try:
        logger.info(f"Téléchargement: {direct_url}")
        
        # Tentative avec URL directe
        response = requests.get(direct_url, timeout=30)
        
        if response.status_code == 404:
            # Alternative: page de la ligue
            logger.info("URL directe non disponible, tentative alternative...")
            response = requests.get(
                url,
                params={'season': season, 'league': league},
                timeout=30
            )
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Téléchargement réussi: {output_path}")
            return str(output_path)
        else:
            logger.error(f"Échec téléchargement: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Erreur téléchargement: {str(e)}")
        return None


def update_polymarket_data(
    output_dir: str = 'data/raw',
    leagues: Optional[List[str]] = None,
    max_markets: int = 50
) -> int:
    """
    Rafraîchit les données Polymarket pour les ligues spécifiées.
    
    Args:
        output_dir: Répertoire de sauvegarde
        leagues: Liste des ligues (défaut: principales ligues européennes)
        max_markets: Nombre maximum de marchés à récupérer
        
    Returns:
        Nombre de marchés récupérés
    """
    if leagues is None:
        leagues = ['E0', 'SP1', 'I1', 'D1', 'F1']  # Top 5 ligues
    
    client = PolymarketClient()
    total_markets = 0
    output_path = Path(output_dir) / 'polymarket'
    output_path.mkdir(parents=True, exist_ok=True)
    
    for league in leagues:
        league_name = FootballDataLoader.LEAGUE_MAPPING.get(league, league)
        
        try:
            markets = client.search_markets(
                query=league_name,
                category="Sports",
                limit=max_markets // len(leagues)
            )
            
            if markets:
                # Sauvegarde par ligue
                file_path = output_path / f"polymarket_{league}_{datetime.now().strftime('%Y%m%d')}.json"
                
                markets_data = []
                for market in markets:
                    markets_data.append({
                        'event_id': market.event_id,
                        'title': market.title,
                        'outcomes': [
                            {'name': o.name, 'price': o.price, 'volume': o.volume}
                            for o in market.outcomes
                        ],
                        'volume': market.volume,
                        'liquidity': market.liquidity,
                        'close_date': market.close_date.isoformat() if market.close_date else None,
                        'url': market.url
                    })
                
                with open(file_path, 'w') as f:
                    json.dump(markets_data, f, indent=2)
                
                total_markets += len(markets)
                logger.info(f"{league}: {len(markets)} marchés récupérés")
                
        except Exception as e:
            logger.warning(f"Erreur Polymarket pour {league}: {str(e)}")
    
    logger.info(f"Total Polymarket: {total_markets} marchés")
    return total_markets


# =============================================================================
# CLASSE NotificationService
# =============================================================================

class NotificationService:
    """
    Service de notification pour les alertes du pipeline.
    
    Supporte :
    - Console (logging)
    - Webhook Telegram
    - Webhook Discord
    - Email (via SMTP)
    """
    
    def __init__(
        self,
        enabled_channels: Optional[List[str]] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        discord_webhook_url: Optional[str] = None,
        smtp_config: Optional[Dict[str, str]] = None
    ):
        """
        Initialise le service de notification.
        
        Args:
            enabled_channels: Liste des canaux activés ['console', 'telegram', 'discord', 'email']
            telegram_bot_token: Token du bot Telegram
            telegram_chat_id: ID du chat Telegram
            discord_webhook_url: URL du webhook Discord
            smtp_config: Configuration SMTP {'host', 'port', 'user', 'password', 'from', 'to'}
        """
        self.enabled_channels = enabled_channels or ['console']
        self.telegram_bot_token = telegram_bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.discord_webhook_url = discord_webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.smtp_config = smtp_config or {
            'host': os.getenv('SMTP_HOST'),
            'port': os.getenv('SMTP_PORT', '587'),
            'user': os.getenv('SMTP_USER'),
            'password': os.getenv('SMTP_PASSWORD'),
            'from': os.getenv('SMTP_FROM'),
            'to': os.getenv('SMTP_TO')
        }
        
        logger.info(f"NotificationService initialized - Channels: {self.enabled_channels}")
    
    def send(self, message: str, title: str = "Predictic Match", level: str = "info"):
        """
        Envoie une notification sur tous les canaux activés.
        
        Args:
            message: Contenu du message
            title: Titre de la notification
            level: Niveau ('info', 'warning', 'error', 'success')
        """
        for channel in self.enabled_channels:
            try:
                if channel == 'console':
                    self._send_console(message, title, level)
                elif channel == 'telegram':
                    self._send_telegram(message, title, level)
                elif channel == 'discord':
                    self._send_discord(message, title, level)
                elif channel == 'email':
                    self._send_email(message, title, level)
            except Exception as e:
                logger.error(f"Erreur notification {channel}: {str(e)}")
    
    def _send_console(self, message: str, title: str, level: str):
        """Envoie vers la console."""
        emoji = {'info': 'ℹ️', 'warning': '⚠️', 'error': '❌', 'success': '✅'}.get(level, 'ℹ️')
        log_func = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}.get(level, logger.info)
        log_func(f"{emoji} [{title}] {message}")
    
    def _send_telegram(self, message: str, title: str, level: str):
        """Envoie vers Telegram."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return
        
        emoji = {'info': 'ℹ️', 'warning': '⚠️', 'error': '❌', 'success': '✅'}.get(level, 'ℹ️')
        text = f"*{emoji} {title}*\n\n{message}"
        
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        data = {
            'chat_id': self.telegram_chat_id,
            'text': text,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
    
    def _send_discord(self, message: str, title: str, level: str):
        """Envoie vers Discord."""
        if not self.discord_webhook_url:
            return
        
        color = {'info': 3447003, 'warning': 16766720, 'error': 15158332, 'success': 3066993}.get(level, 3447003)
        
        embed = {
            'title': title,
            'description': message,
            'color': color,
            'timestamp': datetime.now().isoformat(),
            'footer': {'text': 'Predictic Match Automation'}
        }
        
        data = {'embeds': [embed]}
        
        response = requests.post(self.discord_webhook_url, json=data, timeout=10)
        response.raise_for_status()
    
    def _send_email(self, message: str, title: str, level: str):
        """Envoie par email."""
        if not all(self.smtp_config.values()):
            return
        
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        msg = MIMEMultipart()
        msg['From'] = self.smtp_config['from']
        msg['To'] = self.smtp_config['to']
        msg['Subject'] = f"[Predictic Match] {title}"
        
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP(self.smtp_config['host'], int(self.smtp_config['port']))
        server.starttls()
        server.login(self.smtp_config['user'], self.smtp_config['password'])
        server.send_message(msg)
        server.quit()
    
    def notify_pipeline_result(self, result: PipelineResult):
        """
        Envoie une notification avec le résultat du pipeline.
        
        Args:
            result: Résultat du pipeline
        """
        if result.status == PipelineStatus.COMPLETED:
            level = 'success'
            title = "✅ Pipeline Complété"
        elif result.status == PipelineStatus.FAILED:
            level = 'error'
            title = "❌ Pipeline Échoué"
        else:
            level = 'warning'
            title = "⚠️ Pipeline Partiel"
        
        message = (
            f"**Ligue:** {result.league}\n"
            f"**Saison:** {result.season}\n"
            f"**Matchs traités:** {result.matches_processed}\n"
            f"**Prédictions:** {result.predictions_generated}\n"
            f"**Marchés Polymarket:** {result.polymarket_markets}\n"
        )
        
        if result.model_accuracy:
            message += f"**Accuracy modèle:** {result.model_accuracy:.2%}\n"
        
        if result.error_message:
            message += f"\n**Erreur:** {result.error_message}"
        
        self.send(message, title, level)


# =============================================================================
# SCHEDULING
# =============================================================================

def schedule_predictions(
    league: str = 'E0',
    seasons: Optional[List[int]] = None,
    frequency: str = 'daily',
    time: str = '06:00',
    output_dir: str = 'output/predictions',
    enable_notifications: bool = True
):
    """
    Planifie l'exécution automatique du pipeline.
    
    Args:
        league: Code de la ligue
        seasons: Liste des saisons (défaut: saison actuelle)
        frequency: Fréquence ('daily', 'weekly', 'hourly')
        time: Heure d'exécution (format HH:MM)
        output_dir: Répertoire de sortie
        enable_notifications: Active les notifications
    """
    try:
        import schedule
    except ImportError:
        logger.error("Module 'schedule' non installé. pip install schedule")
        return
    
    if seasons is None:
        now = datetime.now()
        seasons = [now.year - 1 if now.month < 8 else now.year]
    
    notification_service = NotificationService() if enable_notifications else None
    
    def run_pipeline():
        """Fonction exécutée par le scheduler."""
        logger.info(f"Démarrage pipeline planifié - {league}")
        
        for season in seasons:
            pipeline = PredictionPipeline(
                league=league,
                season=season,
                output_dir=output_dir
            )
            
            result = pipeline.run(retrain=False)
            
            if notification_service:
                notification_service.notify_pipeline_result(result)
        
        logger.info("Pipeline planifié terminé")
    
    # Configuration du schedule
    hour, minute = map(int, time.split(':'))
    
    if frequency == 'daily':
        schedule.every().day.at(time).do(run_pipeline)
        logger.info(f"Pipeline planifié tous les jours à {time}")
    
    elif frequency == 'weekly':
        schedule.every().monday.at(time).do(run_pipeline)
        logger.info(f"Pipeline planifié chaque lundi à {time}")
    
    elif frequency == 'hourly':
        schedule.every().hour.do(run_pipeline)
        logger.info(f"Pipeline planifié toutes les heures")
    
    else:
        logger.error(f"Fréquence inconnue: {frequency}")
        return
    
    # Exécution en boucle (bloquant)
    logger.info("Scheduler démarré. Appuyez sur Ctrl+C pour arrêter.")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Vérifie chaque minute


# =============================================================================
# POINT D'ENTRÉE CLI
# =============================================================================

def main():
    """Point d'entrée CLI pour l'automation."""
    parser = argparse.ArgumentParser(
        description="Predictic Match - Automation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python -m src.automation --league E0 --season 2024
  python -m src.automation --league SP1 --output output/la_liga
  python -m src.automation --fetch --league E0 --season 2024
  python -m src.automation --polymarket --leagues E0,SP1,I1
        """
    )
    
    # Arguments principaux
    parser.add_argument('--league', '-l', default='E0',
                        help='Code de la ligue (E0, SP1, I1, D1, F1)')
    parser.add_argument('--season', '-s', type=int, default=None,
                        help='Saison (ex: 2024 pour 2024-2025)')
    parser.add_argument('--output', '-o', default='output/predictions',
                        help='Répertoire de sortie')
    parser.add_argument('--data-dir', '-d', default='data/raw',
                        help='Répertoire des données')
    
    # Actions
    parser.add_argument('--fetch', action='store_true',
                        help='Télécharger les dernières données')
    parser.add_argument('--polymarket', action='store_true',
                        help='Rafraîchir les données Polymarket')
    parser.add_argument('--run', action='store_true',
                        help='Exécuter le pipeline complet')
    parser.add_argument('--schedule', action='store_true',
                        help='Planifier l\'exécution automatique')
    
    # Options
    parser.add_argument('--retrain', action='store_true',
                        help='Forcer le réentraînement du modèle')
    parser.add_argument('--frequency', default='daily',
                        choices=['daily', 'weekly', 'hourly'],
                        help='Fréquence de planification')
    parser.add_argument('--time', default='06:00',
                        help='Heure d\'exécution (HH:MM)')
    parser.add_argument('--notify', action='store_true',
                        help='Activer les notifications')
    parser.add_argument('--leagues', default=None,
                        help='Liste des ligues (séparées par des virgules)')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Mode verbeux')
    parser.add_argument('--log-file', default=None,
                        help='Fichier de log')
    
    args = parser.parse_args()
    
    # Configuration logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    
    # Exécution
    if args.fetch:
        leagues = args.leagues.split(',') if args.leagues else [args.league]
        for league in leagues:
            fetch_latest_data(
                league=league,
                season=args.season or datetime.now().year,
                output_dir=args.data_dir,
                force=True
            )
    
    elif args.polymarket:
        leagues = args.leagues.split(',') if args.leagues else None
        update_polymarket_data(
            output_dir=args.data_dir,
            leagues=leagues
        )
    
    elif args.schedule:
        leagues = args.leagues.split(',') if args.leagues else [args.league]
        for league in leagues:
            schedule_predictions(
                league=league,
                seasons=[args.season] if args.season else None,
                frequency=args.frequency,
                time=args.time,
                output_dir=args.output,
                enable_notifications=args.notify
            )
    
    else:
        # Exécution par défaut: run pipeline
        pipeline = PredictionPipeline(
            league=args.league,
            season=args.season,
            output_dir=args.output,
            data_dir=args.data_dir,
            use_existing_model=not args.retrain
        )
        
        result = pipeline.run(retrain=args.retrain)
        
        print("\n" + "=" * 60)
        print("RÉSULTAT DU PIPELINE")
        print("=" * 60)
        print(f"Statut: {result.status.value}")
        print(f"Ligue: {result.league}")
        print(f"Saison: {result.season}")
        print(f"Matchs traités: {result.matches_processed}")
        print(f"Prédictions: {result.predictions_generated}")
        print(f"Marchés Polymarket: {result.polymarket_markets}")
        if result.model_accuracy:
            print(f"Accuracy modèle: {result.model_accuracy:.2%}")
        if result.error_message:
            print(f"Erreur: {result.error_message}")
        print("=" * 60)
        
        # Notification si activée
        if args.notify:
            notifier = NotificationService()
            notifier.notify_pipeline_result(result)
        
        # Code de sortie
        sys.exit(0 if result.status == PipelineStatus.COMPLETED else 1)


if __name__ == "__main__":
    main()
