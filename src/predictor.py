"""
Script principal de prédiction - Orchestre tout le pipeline Predictic Match
"""
import os
import sys
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Import des modules locaux
from src.data_loader import FootballDataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineer import FeatureEngineer
from src.interpreter import PredictionInterpreter

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('output/predictions/pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class PredicticPredictor:
    """
    Classe principale qui orchestre tout le pipeline de prédiction.
    
    Pipeline:
    1. Chargement des données (FootballDataLoader)
    2. Nettoyage (DataCleaner)
    3. Feature Engineering (FeatureEngineer)
    4. Prédiction (Modèle ML)
    5. Interprétation (PredictionInterpreter)
    6. Export des résultats
    """
    
    OUTCOME_MAPPING = {2: 'H', 1: 'D', 0: 'A'}
    OUTCOME_LABELS = {2: 'Victoire domicile', 1: 'Match nul', 0: 'Victoire extérieur'}
    
    def __init__(
        self,
        data_dir: str = "data/raw",
        model_dir: str = "data/models",
        output_dir: str = "output/predictions"
    ):
        """
        Initialise le pipeline de prédiction.
        
        Args:
            data_dir: Répertoire des données brutes
            model_dir: Répertoire des modèles entraînés
            output_dir: Répertoire de sortie des prédictions
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        
        # Initialiser les composants
        self.loader = FootballDataLoader(data_dir)
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.interpreter = PredictionInterpreter()
        
        # Modèle (chargé ou entraîné)
        self.model = None
        self.feature_columns = None
        
        # Créer les répertoires si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info("Pipeline Predictic initialisé")
    
    def load_or_train_model(
        self,
        leagues: List[str] = ['E0'],
        seasons: List[str] = None,
        force_retrain: bool = False
    ) -> bool:
        """
        Charge un modèle existant ou en entraîne un nouveau.
        
        Args:
            leagues: Ligues à utiliser pour l'entraînement
            seasons: Saisons à utiliser (None = toutes disponibles)
            force_retrain: Force le réentraînement même si modèle existe
            
        Returns:
            True si succès, False sinon
        """
        model_path = os.path.join(self.model_dir, 'predictic_model.pkl')
        
        # Essayer de charger le modèle existant
        if os.path.exists(model_path) and not force_retrain:
            logger.info(f"Chargement du modèle depuis {model_path}")
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.feature_columns = model_data['feature_columns']
                logger.info("Modèle chargé avec succès")
                return True
            except Exception as e:
                logger.warning(f"Erreur de chargement: {e}. Réentraînement nécessaire.")
        
        # Entraîner un nouveau modèle
        logger.info("Entraînement d'un nouveau modèle...")
        return self._train_model(leagues, seasons)
    
    def _train_model(self, leagues: List[str], seasons: List[str]) -> bool:
        """
        Entraîne un nouveau modèle.
        
        Note: Cette méthode utilise une implémentation simplifiée.
        Pour un vrai entraînement, utiliser src/model.py complet.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier, VotingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            import xgboost as xgb
            
            # Charger et préparer les données
            all_data = []
            for league in leagues:
                if seasons:
                    df = self.loader.load_multiple_seasons(league, seasons)
                else:
                    # Charger toutes les saisons disponibles
                    available = self._get_available_seasons(league)
                    if available:
                        df = self.loader.load_multiple_seasons(league, available)
                    else:
                        continue
                all_data.append(df)
            
            if not all_data:
                logger.error("Aucune donnée disponible pour l'entraînement")
                return False
            
            df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Données chargées: {len(df)} matchs")
            
            # Nettoyer
            df = self.cleaner.clean(df)
            
            # Features
            df = self.engineer.create_all_features(df)
            
            # Préparer X et y
            feature_cols = self.engineer.get_feature_columns()
            
            # Supprimer les colonnes non numériques
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = numeric_cols
            
            X = df[numeric_cols].fillna(0)
            y = df['FTR_encoded'].fillna(1).astype(int)
            
            # Split temporel (80% train, 20% test)
            split_idx = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Modèles
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            lr = LogisticRegression(max_iter=1000, random_state=42)
            
            # Ensemble
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('xgb', xgb_model), ('lr', lr)],
                voting='soft'
            )
            
            # Entraînement
            logger.info("Entraînement du modèle...")
            ensemble.fit(X_train_scaled, y_train)
            
            # Évaluation
            train_acc = ensemble.score(X_train_scaled, y_train)
            test_acc = ensemble.score(X_test_scaled, y_test)
            logger.info(f"Accuracy Train: {train_acc:.3f}, Test: {test_acc:.3f}")
            
            # Sauvegarder
            self.model = {
                'model': ensemble,
                'scaler': scaler,
                'feature_columns': self.feature_columns
            }
            
            model_path = os.path.join(self.model_dir, 'predictic_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info(f"Modèle sauvegardé: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur d'entraînement: {e}")
            return False
    
    def _get_available_seasons(self, league: str) -> List[str]:
        """Récupère les saisons disponibles pour une ligue."""
        # Implémentation simplifiée - à adapter selon vos fichiers
        available = []
        for year in range(18, 26):  # 2018-2025
            season = f"{year}-{str(year+1)[-2:]}"
            filepath = os.path.join(self.data_dir, f"{league}_{season}.csv")
            if os.path.exists(filepath):
                available.append(season)
        return available
    
    def predict_match(
        self,
        match_data: Dict,
        recent_form: Optional[Dict] = None
    ) -> Dict:
        """
        Prédit le résultat d'un match spécifique.
        
        Args:
            match_data: Données du match (équipes, cotes, etc.)
            recent_form: Forme récente des équipes (optionnel)
            
        Returns:
            Dictionnaire avec prédiction et interprétation
        """
        if self.model is None:
            raise ValueError("Modèle non chargé. Appeler load_or_train_model() d'abord.")
        
        # Créer un DataFrame temporaire pour le feature engineering
        # Note: Pour une vraie prédiction, il faudrait l'historique complet
        # pour calculer les features temporelles (ELO, rolling stats, H2H)
        
        df_temp = pd.DataFrame([match_data])
        
        # Features de base depuis les cotes
        if all(k in match_data for k in ['B365H', 'B365D', 'B365A']):
            odds = [match_data['B365H'], match_data['B365D'], match_data['B365A']]
            implied = 1.0 / pd.Series(odds)
            prob_sum = implied.sum()
            df_temp['implied_prob_home'] = implied.iloc[0] / prob_sum
            df_temp['implied_prob_draw'] = implied.iloc[1] / prob_sum
            df_temp['implied_prob_away'] = implied.iloc[2] / prob_sum
        
        # Ajouter des features par défaut si manquantes
        for col in self.feature_columns:
            if col not in df_temp.columns:
                df_temp[col] = 0
        
        # Préparation
        X = df_temp[self.feature_columns].fillna(0)
        X_scaled = self.model['scaler'].transform(X)
        
        # Prédiction
        probabilities = self.model['model'].predict_proba(X_scaled)[0]
        predicted_class = self.model['model'].predict(X_scaled)[0]
        
        # Construire les probabilités par classe
        classes = self.model['model'].classes_
        proba_dict = {}
        for cls, prob in zip(classes, probabilities):
            key = ['away', 'draw', 'home'][cls]  # 0=A, 1=D, 2=H
            proba_dict[key] = float(prob)
        
        prediction = {
            'probabilities': proba_dict,
            'predicted_class': int(predicted_class),
            'predicted_outcome': self.OUTCOME_MAPPING.get(predicted_class, 'Unknown'),
            'confidence': float(max(probabilities))
        }
        
        # Interprétation
        interpretation = self.interpreter.interpret_prediction(
            match_data=match_data,
            model_prediction=prediction,
            features=recent_form or {}
        )
        
        return {
            'match': match_data,
            'prediction': prediction,
            'interpretation': interpretation,
            'report': self.interpreter.generate_report(interpretation)
        }
    
    def predict_batch(
        self,
        matches: List[Dict],
        output_file: Optional[str] = None
    ) -> List[Dict]:
        """
        Prédit les résultats de plusieurs matchs.
        
        Args:
            matches: Liste des données de matchs
            output_file: Fichier de sortie (optionnel)
            
        Returns:
            Liste des prédictions
        """
        results = []
        
        for i, match in enumerate(matches):
            logger.info(f"Prédiction match {i+1}/{len(matches)}: {match.get('HomeTeam')} vs {match.get('AwayTeam')}")
            
            try:
                result = self.predict_match(match)
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur prédiction match {i+1}: {e}")
                results.append({
                    'match': match,
                    'error': str(e)
                })
        
        # Export
        if output_file:
            self._export_predictions(results, output_file)
        
        return results
    
    def _export_predictions(self, predictions: List[Dict], output_file: str):
        """Exporte les prédictions vers un fichier."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_file.endswith('.json'):
            filepath = os.path.join(self.output_dir, output_file)
        else:
            filepath = os.path.join(self.output_dir, f"{output_file}_{timestamp}.json")
        
        # Convertir pour JSON
        export_data = []
        for pred in predictions:
            export_item = {
                'home_team': pred['match'].get('HomeTeam'),
                'away_team': pred['match'].get('AwayTeam'),
                'date': str(pred['match'].get('Date', '')),
                'predicted_outcome': pred['prediction'].get('predicted_outcome'),
                'confidence': pred['prediction'].get('confidence'),
                'probabilities': pred['prediction'].get('probabilities'),
                'recommendation': pred['interpretation'].get('recommendation', {}),
                'value_bet': pred['interpretation'].get('value_bet', {})
            }
            export_data.append(export_item)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Prédictions exportées: {filepath}")
    
    def get_pipeline_status(self) -> Dict:
        """Retourne le statut du pipeline."""
        model_path = os.path.join(self.model_dir, 'predictic_model.pkl')
        
        return {
            'model_loaded': self.model is not None,
            'model_file_exists': os.path.exists(model_path),
            'feature_columns_count': len(self.feature_columns) if self.feature_columns else 0,
            'output_dir': self.output_dir,
            'data_dir': self.data_dir
        }


def main():
    """
    Fonction principale - Exemple d'utilisation du pipeline.
    """
    logger.info("=" * 50)
    logger.info("Predictic Match - Pipeline de Prédiction")
    logger.info("=" * 50)
    
    # Initialiser le pipeline
    predictor = PredicticPredictor(
        data_dir="data/raw",
        model_dir="data/models",
        output_dir="output/predictions"
    )
    
    # Charger ou entraîner le modèle
    success = predictor.load_or_train_model(
        leagues=['E0'],  # Premier League
        seasons=['23-24', '24-25'],
        force_retrain=False
    )
    
    if not success:
        logger.error("Échec du chargement/entraînement du modèle")
        return
    
    # Exemple de prédiction pour un match
    sample_match = {
        'Date': datetime.now(),
        'HomeTeam': 'Manchester City',
        'AwayTeam': 'Liverpool',
        'league': 'E0',
        'B365H': 2.10,
        'B365D': 3.40,
        'B365A': 3.20
    }
    
    # Prédire
    result = predictor.predict_match(
        sample_match,
        recent_form={
            'elo_diff': 50,
            'home_form_5': 2.4,
            'h2h_home_wins': 3,
            'h2h_total': 10
        }
    )
    
    # Afficher le rapport
    print("\n" + "=" * 50)
    print(result['report'])
    print("=" * 50)
    
    # Statistiques
    stats = predictor.interpreter.get_summary_statistics()
    logger.info(f"Statistiques: {stats}")
    
    # Statut du pipeline
    status = predictor.get_pipeline_status()
    logger.info(f"Statut du pipeline: {status}")


if __name__ == "__main__":
    main()
