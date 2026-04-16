"""
Module de modélisation ML pour la classification sportive (H/D/A)
Objectif : >50% accuracy sur 3 classes avec validation temporelle
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix

# Import optionnel de XGBoost (peut nécessiter libomp sur macOS)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    xgb = None


def prepare_model_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_col: str = 'FTR_encoded',
    test_size: float = 0.2,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Prépare les données pour l'entraînement avec TimeSeriesSplit.
    
    Args:
        df: DataFrame avec features et target
        feature_columns: Liste des colonnes features
        target_col: Nom de la colonne cible
        test_size: Proportion pour le test set final
        n_splits: Nombre de splits pour TimeSeriesSplit
        
    Returns:
        Dictionnaire contenant X, y, scaler, tscv, et indices train/test
    """
    # Nettoyage des données
    df_clean = df.dropna(subset=feature_columns + [target_col]).copy()
    
    # Extraction features/target
    X = df_clean[feature_columns].values
    y = df_clean[target_col].values
    
    # Split temporel (les dernières données pour le test)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # TimeSeriesSplit pour validation croisée
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'y_train_raw': y_train,
        'y_test_raw': y_test,
        'scaler': scaler,
        'tscv': tscv,
        'feature_columns': feature_columns,
        'split_idx': split_idx,
        'train_dates': df_clean['Date'].iloc[:split_idx].values if 'Date' in df_clean.columns else None,
        'test_dates': df_clean['Date'].iloc[split_idx:].values if 'Date' in df_clean.columns else None
    }


def train_and_evaluate(
    data: Dict[str, Any],
    models: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Entraîne et évalue plusieurs modèles avec validation croisée temporelle.
    
    Args:
        data: Dictionnaire de données préparées par prepare_model_data()
        models: Dictionnaire de modèles personnalisés (optionnel)
        
    Returns:
        Dictionnaire avec résultats d'évaluation pour chaque modèle
    """
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    tscv = data['tscv']
    
    # Définition des modèles
    if models is None:
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                solver='lbfgs'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }
        # Ajout de XGBoost si disponible
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
        else:
            print("Note: XGBoost non disponible (libomp manquant sur macOS)")
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Entraînement: {name}")
        print('='*50)
        
        # Validation croisée temporelle
        cv_scores = []
        cv_logloss = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            
            cv_scores.append(accuracy_score(y_val, y_pred))
            cv_logloss.append(log_loss(y_val, y_proba))
        
        # Entraînement final sur tout le train set
        model.fit(X_train, y_train)
        
        # Évaluation sur le test set
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)
        
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_logloss = log_loss(y_test, y_proba_test)
        
        results[name] = {
            'model': model,
            'cv_accuracy_mean': np.mean(cv_scores),
            'cv_accuracy_std': np.std(cv_scores),
            'cv_logloss_mean': np.mean(cv_logloss),
            'cv_logloss_std': np.std(cv_logloss),
            'test_accuracy': test_accuracy,
            'test_logloss': test_logloss,
            'y_pred_test': y_pred_test,
            'y_proba_test': y_proba_test
        }
        
        print(f"CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        print(f"CV Log Loss: {np.mean(cv_logloss):.4f} (+/- {np.std(cv_logloss):.4f})")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Log Loss: {test_logloss:.4f}")
        
        # Rapport de classification
        print("\nClassification Report:")
        target_names = ['Away (0)', 'Draw (1)', 'Home (2)']
        print(classification_report(y_test, y_pred_test, target_names=target_names))
    
    return results


def build_ensemble(
    data: Dict[str, Any],
    model_results: Dict[str, Dict[str, Any]],
    voting: str = 'soft',
    use_weights: bool = True
) -> Dict[str, Any]:
    """
    Construit un VotingClassifier avec soft voting et poids optimisés.
    
    Args:
        data: Données préparées
        model_results: Résultats de train_and_evaluate()
        voting: Type de voting ('soft' ou 'hard')
        use_weights: Utiliser des poids basés sur la performance
        
    Returns:
        Dictionnaire avec le modèle ensemble et ses résultats
    """
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    tscv = data['tscv']
    
    # Sélection des modèles pour l'ensemble (top performers)
    estimators = []
    weights = []
    
    for name, result in model_results.items():
        model = result['model']
        estimators.append((name, model))
        
        if use_weights:
            # Poids basé sur l'accuracy CV
            weight = result['cv_accuracy_mean']
            weights.append(weight)
    
    # Normalisation des poids
    if use_weights and weights:
        weights = np.array(weights)
        weights = weights / weights.sum()
        print(f"\nPoids des modèles: {dict(zip([n for n, _ in estimators], weights))}")
    
    # Création du VotingClassifier
    if use_weights and voting == 'soft':
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights.tolist()
        )
    else:
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting
        )
    
    # Validation croisée temporelle
    cv_scores = []
    cv_logloss = []
    
    print(f"\n{'='*50}")
    print("Entraînement: Voting Ensemble")
    print('='*50)
    
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        ensemble.fit(X_tr, y_tr)
        y_pred = ensemble.predict(X_val)
        y_proba = ensemble.predict_proba(X_val)
        
        cv_scores.append(accuracy_score(y_val, y_pred))
        cv_logloss.append(log_loss(y_val, y_proba))
    
    # Entraînement final
    ensemble.fit(X_train, y_train)
    
    # Évaluation sur le test set
    y_pred_test = ensemble.predict(X_test)
    y_proba_test = ensemble.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_logloss = log_loss(y_test, y_proba_test)
    
    print(f"CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print(f"CV Log Loss: {np.mean(cv_logloss):.4f} (+/- {np.std(cv_logloss):.4f})")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Log Loss: {test_logloss:.4f}")
    
    print("\nClassification Report:")
    target_names = ['Away (0)', 'Draw (1)', 'Home (2)']
    print(classification_report(y_test, y_pred_test, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))
    
    return {
        'model': ensemble,
        'cv_accuracy_mean': np.mean(cv_scores),
        'cv_accuracy_std': np.std(cv_scores),
        'cv_logloss_mean': np.mean(cv_logloss),
        'cv_logloss_std': np.std(cv_logloss),
        'test_accuracy': test_accuracy,
        'test_logloss': test_logloss,
        'y_pred_test': y_pred_test,
        'y_proba_test': y_proba_test,
        'estimators': estimators,
        'weights': weights.tolist() if use_weights else None
    }


class WalkForwardBacktest:
    """
    Backtesting temporel correct avec fenêtre glissante.
    Simule un déploiement en production avec réentraînement périodique.
    """
    
    def __init__(
        self,
        initial_train_size: int = 500,
        step_size: int = 50,
        retrain_every: int = 100,
        model_class: Any = None,
        model_params: Optional[Dict] = None
    ):
        """
        Initialise le backtest walk-forward.
        
        Args:
            initial_train_size: Taille initiale du training set
            step_size: Nombre d'échantillons à chaque itération
            retrain_every: Réentraîner tous les X pas
            model_class: Classe du modèle à utiliser
            model_params: Paramètres du modèle
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.retrain_every = retrain_every
        self.model_class = model_class or RandomForestClassifier
        self.model_params = model_params or {
            'n_estimators': 200,
            'max_depth': 15,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        self.results = []
        self.scaler = StandardScaler()
        
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Exécute le backtest walk-forward.
        
        Args:
            X: Features (déjà triées temporellement)
            y: Target
            dates: Dates associées (optionnel)
            
        Returns:
            Dictionnaire avec tous les résultats du backtest
        """
        n_samples = len(X)
        
        if self.initial_train_size >= n_samples:
            raise ValueError("initial_train_size doit être < nombre d'échantillons")
        
        all_predictions = []
        all_probabilities = []
        all_actuals = []
        all_dates = []
        
        train_start = 0
        train_end = self.initial_train_size
        test_start = train_end
        test_end = test_start + self.step_size
        
        iteration = 0
        model = None
        
        print(f"\n{'='*60}")
        print("WALK-FORWARD BACKTEST")
        print('='*60)
        print(f"Initial train size: {self.initial_train_size}")
        print(f"Step size: {self.step_size}")
        print(f"Retrain every: {self.retrain_every} iterations")
        print(f"Total samples: {n_samples}")
        print('='*60)
        
        while test_end <= n_samples:
            iteration += 1
            
            # Données d'entraînement et de test
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # Standardisation (fit sur train, transform sur test)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Réentraînement si nécessaire
            if model is None or iteration % self.retrain_every == 0:
                model = self.model_class(**self.model_params)
                model.fit(X_train_scaled, y_train)
                print(f"\n[Itération {iteration}] Modèle réentraîné")
            
            # Prédictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)
            
            # Stockage des résultats
            all_predictions.extend(y_pred)
            all_probabilities.extend(y_proba)
            all_actuals.extend(y_test)
            
            if dates is not None:
                all_dates.extend(dates[test_start:test_end])
            
            # Métriques de cette fenêtre
            window_accuracy = accuracy_score(y_test, y_pred)
            window_logloss = log_loss(y_test, y_proba)
            
            self.results.append({
                'iteration': iteration,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'window_accuracy': window_accuracy,
                'window_logloss': window_logloss,
                'n_train': len(X_train),
                'n_test': len(X_test)
            })
            
            print(f"  Train: [{train_start}:{train_end}] ({len(X_train)}), "
                  f"Test: [{test_start}:{test_end}] ({len(X_test)}) - "
                  f"Accuracy: {window_accuracy:.4f}")
            
            # Décalage temporel
            train_end += self.step_size
            test_start = train_end
            test_end = test_start + self.step_size
        
        # Résultats agrégés
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        all_probabilities = np.array(all_probabilities)
        
        overall_accuracy = accuracy_score(all_actuals, all_predictions)
        overall_logloss = log_loss(all_actuals, all_probabilities)
        
        print(f"\n{'='*60}")
        print("RÉSULTATS GLOBAUX DU BACKTEST")
        print('='*60)
        print(f"Nombre d'itérations: {iteration}")
        print(f"Accuracy moyenne: {np.mean([r['window_accuracy'] for r in self.results]):.4f}")
        print(f"Accuracy std: {np.std([r['window_accuracy'] for r in self.results]):.4f}")
        print(f"Accuracy globale: {overall_accuracy:.4f}")
        print(f"Log Loss global: {overall_logloss:.4f}")
        
        target_names = ['Away (0)', 'Draw (1)', 'Home (2)']
        print("\nClassification Report:")
        print(classification_report(all_actuals, all_predictions, target_names=target_names))
        
        return {
            'overall_accuracy': overall_accuracy,
            'overall_logloss': overall_logloss,
            'mean_window_accuracy': np.mean([r['window_accuracy'] for r in self.results]),
            'std_window_accuracy': np.std([r['window_accuracy'] for r in self.results]),
            'iterations': iteration,
            'all_predictions': all_predictions,
            'all_probabilities': all_probabilities,
            'all_actuals': all_actuals,
            'all_dates': all_dates if all_dates else None,
            'window_results': self.results
        }


def save_model(model: Any, filepath: str, metadata: Optional[Dict] = None) -> str:
    """
    Sauvegarde un modèle et ses métadonnées avec pickle.
    
    Args:
        model: Modèle à sauvegarder
        filepath: Chemin du fichier de sauvegarde
        metadata: Métadonnées additionnelles (optionnel)
        
    Returns:
        Chemin du fichier sauvegardé
    """
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Préparer les données à sauvegarder
    save_data = {
        'model': model,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"\nModèle sauvegardé: {filepath}")
    return filepath


def load_model(filepath: str) -> Tuple[Any, Dict]:
    """
    Charge un modèle et ses métadonnées depuis un fichier pickle.
    
    Args:
        filepath: Chemin du fichier à charger
        
    Returns:
        Tuple (modèle, métadonnées)
    """
    with open(filepath, 'rb') as f:
        save_data = pickle.load(f)
    
    model = save_data['model']
    metadata = save_data.get('metadata', {})
    
    print(f"Modèle chargé: {filepath}")
    print(f"Sauvegardé le: {save_data.get('saved_at', 'inconnu')}")
    
    return model, metadata


# Exemple d'utilisation
if __name__ == "__main__":
    # Test rapide avec données synthétiques
    print("Test du module model.py")
    print("="*50)
    
    # Création de données de test
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)  # 3 classes: 0, 1, 2
    
    # Préparation des données
    data = {
        'X_train': X[:800],
        'X_test': X[800:],
        'y_train': y[:800],
        'y_test': y[800:],
        'tscv': TimeSeriesSplit(n_splits=5)
    }
    
    # Entraînement et évaluation
    results = train_and_evaluate(data)
    
    # Construction de l'ensemble
    ensemble_results = build_ensemble(data, results)
    
    # Walk-forward backtest
    backtest = WalkForwardBacktest(
        initial_train_size=500,
        step_size=50,
        retrain_every=3
    )
    backtest_results = backtest.run(X, y)
    
    print("\n" + "="*50)
    print("Test terminé avec succès!")
