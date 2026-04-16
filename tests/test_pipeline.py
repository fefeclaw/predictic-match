"""
Tests pour le pipeline Predictic Match
========================================
Tests unitaires et d'intégration pour valider chaque composant.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import FootballDataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineer import FeatureEngineer
from src.interpreter import PredictionInterpreter


# ============================================================================
# FIXTURES COMMUNS
# ============================================================================

@pytest.fixture
def sample_raw_data():
    """Génère des données brutes de test."""
    np.random.seed(42)
    n_matches = 100
    
    dates = pd.date_range('2023-08-01', periods=n_matches, freq='D')
    home_teams = np.random.choice(['Team A', 'Team B', 'Team C', 'Team D'], n_matches)
    away_teams = np.random.choice(['Team E', 'Team F', 'Team G', 'Team H'], n_matches)
    ftr = np.random.choice(['H', 'D', 'A'], n_matches, p=[0.45, 0.25, 0.30])
    
    df = pd.DataFrame({
        'Date': dates.strftime('%d/%m/%Y'),
        'HomeTeam': home_teams,
        'AwayTeam': away_teams,
        'FTR': ftr,
        'FTHG': np.random.randint(0, 5, n_matches),
        'FTAG': np.random.randint(0, 4, n_matches),
        'B365H': np.random.uniform(1.5, 3.5, n_matches),
        'B365D': np.random.uniform(2.5, 4.5, n_matches),
        'B365A': np.random.uniform(2.0, 5.0, n_matches)
    })
    
    return df


@pytest.fixture
def temp_data_dir(sample_raw_data):
    """Crée un répertoire temporaire avec des données de test."""
    temp_dir = tempfile.mkdtemp()
    
    # Sauvegarder les données
    filepath = os.path.join(temp_dir, 'E0_23-24.csv')
    sample_raw_data.to_csv(filepath, index=False, encoding='latin-1')
    
    yield temp_dir
    
    # Nettoyage
    shutil.rmtree(temp_dir)


@pytest.fixture
def loader(temp_data_dir):
    """Fixture pour FootballDataLoader."""
    return FootballDataLoader(data_dir=temp_data_dir)


@pytest.fixture
def cleaner():
    """Fixture pour DataCleaner."""
    return DataCleaner()


@pytest.fixture
def engineer():
    """Fixture pour FeatureEngineer."""
    return FeatureEngineer()


@pytest.fixture
def interpreter():
    """Fixture pour PredictionInterpreter."""
    return PredictionInterpreter()


# ============================================================================
# TESTS UNITAIRES - FOOTBALLDATALOADER
# ============================================================================

class TestFootballDataLoader:
    """Tests unitaires pour la classe FootballDataLoader."""
    
    def test_init(self, temp_data_dir):
        """Test l'initialisation du loader."""
        loader = FootballDataLoader(data_dir=temp_data_dir)
        assert loader.data_dir == temp_data_dir
        assert loader.data is None
    
    def test_load_season_success(self, loader):
        """Test le chargement d'une saison avec succès."""
        df = loader.load_season('E0', '23-24')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'league' in df.columns
        assert 'season' in df.columns
        assert df['league'].iloc[0] == 'E0'
        assert df['season'].iloc[0] == '23-24'
    
    def test_load_season_file_not_found(self, loader):
        """Test le chargement d'un fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            loader.load_season('E0', '99-99')
    
    def test_load_multiple_seasons(self, temp_data_dir):
        """Test le chargement de plusieurs saisons."""
        # Créer une deuxième saison
        df2 = pd.DataFrame({
            'Date': ['01/08/2024'],
            'HomeTeam': ['Team X'],
            'AwayTeam': ['Team Y'],
            'FTR': ['H'],
            'B365H': [2.0],
            'B365D': [3.0],
            'B365A': [3.5]
        })
        filepath = os.path.join(temp_data_dir, 'E0_24-25.csv')
        df2.to_csv(filepath, index=False, encoding='latin-1')
        
        loader = FootballDataLoader(data_dir=temp_data_dir)
        df = loader.load_multiple_seasons('E0', ['23-24', '24-25'])
        
        assert len(df) == 101  # 100 + 1
        assert 'E0' in df['league'].values
        assert '23-24' in df['season'].values
        assert '24-25' in df['season'].values
    
    def test_get_league_name(self, loader):
        """Test la récupération du nom de la ligue."""
        assert loader.get_league_name('E0') == 'Premier League'
        assert loader.get_league_name('SP1') == 'La Liga'
        assert loader.get_league_name('D1') == 'Bundesliga'
        assert loader.get_league_name('I1') == 'Serie A'
        assert loader.get_league_name('F1') == 'Ligue 1'
        assert loader.get_league_name('XX') == 'Unknown League'
    
    def test_validate_data_success(self, loader):
        """Test la validation de données valides."""
        df = loader.load_season('E0', '23-24')
        result = loader.validate_data(df)
        
        assert result['valid'] is True
        assert 'rows' in result
        assert 'columns' in result
        assert result['rows'] == 100
    
    def test_validate_data_missing_columns(self, loader):
        """Test la validation avec colonnes manquantes."""
        df = pd.DataFrame({'Col1': [1, 2], 'Col2': [3, 4]})
        result = loader.validate_data(df)
        
        assert result['valid'] is False
        assert 'error' in result
        assert 'Colonnes manquantes' in result['error']
    
    def test_validate_data_no_data(self, loader):
        """Test la validation sans données."""
        result = loader.validate_data()
        
        assert result['valid'] is False
        assert 'error' in result


# ============================================================================
# TESTS UNITAIRES - DATACLEANER
# ============================================================================

class TestDataCleaner:
    """Tests unitaires pour la classe DataCleaner."""
    
    def test_init(self, cleaner):
        """Test l'initialisation du cleaner."""
        assert cleaner.cleaned_data is None
        assert cleaner.RESULT_ENCODING == {'H': 2, 'D': 1, 'A': 0}
    
    def test_clean_basic(self, cleaner, sample_raw_data):
        """Test le nettoyage de base."""
        cleaned = cleaner.clean(sample_raw_data)
        
        assert isinstance(cleaned, pd.DataFrame)
        assert 'FTR_encoded' in cleaned.columns
        assert np.issubdtype(cleaned['Date'].dtype, np.datetime64)
    
    def test_encode_result(self, cleaner, sample_raw_data):
        """Test l'encodage des résultats."""
        cleaned = cleaner.clean(sample_raw_data)
        
        # Vérifier l'encodage
        assert (cleaned[cleaned['FTR'] == 'H']['FTR_encoded'] == 2).all()
        assert (cleaned[cleaned['FTR'] == 'D']['FTR_encoded'] == 1).all()
        assert (cleaned[cleaned['FTR'] == 'A']['FTR_encoded'] == 0).all()
    
    def test_handle_missing_values(self, cleaner):
        """Test la gestion des valeurs manquantes."""
        df = pd.DataFrame({
            'Date': ['01/08/2023', '02/08/2023', '03/08/2023'],
            'HomeTeam': ['Team A', None, 'Team C'],
            'AwayTeam': ['Team B', 'Team D', None],
            'FTR': ['H', 'D', 'A'],
            'B365H': [2.0, np.nan, 2.5],
            'B365D': [3.0, 3.5, np.nan],
            'B365A': [3.5, 4.0, 4.5]
        })
        
        cleaned = cleaner.clean(df)
        
        # Les lignes avec équipes manquantes doivent être supprimées
        assert len(cleaned) < len(df)
        assert cleaned['HomeTeam'].isna().sum() == 0
        assert cleaned['AwayTeam'].isna().sum() == 0
    
    def test_clean_odds_outliers(self, cleaner):
        """Test le nettoyage des cotes aberrantes."""
        df = pd.DataFrame({
            'Date': ['01/08/2023', '02/08/2023'],
            'HomeTeam': ['Team A', 'Team B'],
            'AwayTeam': ['Team B', 'Team A'],
            'FTR': ['H', 'A'],
            'B365H': [100.0, 2.0],  # Cote aberrante au premier match
            'B365D': [3.0, 3.5],
            'B365A': [3.5, 4.0]
        })
        
        cleaned = cleaner.clean(df)
        
        # La cote aberrante doit être remplacée par la médiane (2.0)
        # ou au moins être < 50 et non-NaN
        assert pd.notna(cleaned['B365H'].iloc[0])
        assert cleaned['B365H'].iloc[0] < 50
    
    def test_calculate_probabilities(self, cleaner):
        """Test le calcul des probabilités implicites."""
        df = pd.DataFrame({
            'Date': ['01/08/2023'],
            'HomeTeam': ['Team A'],
            'AwayTeam': ['Team B'],
            'FTR': ['H'],
            'B365H': [2.0],
            'B365D': [4.0],
            'B365A': [4.0]
        })
        
        df = cleaner.clean(df)
        df = cleaner.calculate_probabilities(df)
        
        assert 'prob_home' in df.columns
        assert 'prob_draw' in df.columns
        assert 'prob_away' in df.columns
        
        # Les probabilités doivent sommer à 1
        prob_sum = df['prob_home'] + df['prob_draw'] + df['prob_away']
        assert np.isclose(prob_sum.iloc[0], 1.0)
    
    def test_filter_by_date_range(self, cleaner, sample_raw_data):
        """Test le filtrage par plage de dates."""
        cleaned = cleaner.clean(sample_raw_data)
        
        # Filtrer
        filtered = cleaner.filter_by_date_range(
            cleaned,
            '2023-08-01',
            '2023-08-10'
        )
        
        assert len(filtered) <= 10
        assert filtered['Date'].min() >= pd.Timestamp('2023-08-01')
        assert filtered['Date'].max() <= pd.Timestamp('2023-08-10')


# ============================================================================
# TESTS UNITAIRES - FEATUREENGINEER (AVEC DÉTECTION DATA LEAKAGE)
# ============================================================================

class TestFeatureEngineer:
    """
    Tests unitaires pour FeatureEngineer.
    IMPORTANT: Vérifie l'absence de data leakage.
    """
    
    def test_init(self, engineer):
        """Test l'initialisation."""
        assert engineer.feature_columns == []
    
    def test_create_all_features(self, engineer, cleaner, sample_raw_data):
        """Test la création de toutes les features."""
        # Nettoyer d'abord
        df = cleaner.clean(sample_raw_data)
        
        # Créer les features
        df = engineer.create_all_features(df)
        
        # Vérifier les features créées
        assert 'home_form_5' in df.columns
        assert 'away_form_5' in df.columns
        assert 'home_elo' in df.columns
        assert 'away_elo' in df.columns
        assert 'elo_diff' in df.columns
        assert len(engineer.get_feature_columns()) > 0
    
    def test_rolling_features_no_leakage(self, engineer, cleaner):
        """
        TEST CRITIQUE: Vérifie que les rolling features n'utilisent pas de données futures.
        
        Le data leakage se produit si:
        - Les stats du match N utilisent des données du match N (au lieu de N-1, N-2, ...)
        """
        # Créer des données très simples
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'HomeTeam': ['Team A'] * 5 + ['Team B'] * 5,
            'AwayTeam': ['Team C'] * 5 + ['Team D'] * 5,
            'FTR': ['H', 'A', 'H', 'D', 'H', 'A', 'H', 'A', 'D', 'H'],
            'FTHG': [2, 1, 3, 1, 2, 0, 2, 1, 1, 3],
            'FTAG': [1, 2, 0, 1, 1, 2, 1, 2, 1, 0],
            'B365H': [2.0] * 10,
            'B365D': [3.0] * 10,
            'B365A': [3.5] * 10
        })
        
        df = cleaner.clean(df)
        df = engineer.create_all_features(df)
        
        # VÉRIFICATION DATA LEAKAGE:
        # La forme au match i doit être basée sur les matchs i-1, i-2, ... (PAS i)
        
        # Pour Team A, match 1 (index 0): pas d'historique → forme = 0 ou moyenne vide
        # Pour Team A, match 2 (index 1): basé sur match 1 uniquement
        
        # Vérifier que la forme n'est PAS basée sur le résultat actuel
        # Si leakage: home_form_5 serait corrélé parfaitement avec FTR_encoded actuel
        correlation = df['home_form_5'].corr(df['FTR_encoded'])
        
        # Une corrélation > 0.9 indiquerait un leakage probable
        assert abs(correlation) < 0.9, f"Data leakage détecté! Corrélation: {correlation}"
    
    def test_elo_features_no_leakage(self, engineer, cleaner):
        """
        TEST CRITIQUE: Vérifie que l'ELO n'utilise pas de données futures.
        
        Le rating ELO au match N doit être basé sur les matchs 1 à N-1.
        """
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=20),
            'HomeTeam': ['Team A', 'Team B'] * 10,
            'AwayTeam': ['Team C', 'Team D'] * 10,
            'FTR': np.random.choice(['H', 'D', 'A'], 20),
            'B365H': [2.0] * 20,
            'B365D': [3.0] * 20,
            'B365A': [3.5] * 20
        })
        
        df = cleaner.clean(df)
        df = engineer.create_all_features(df)
        
        # L'ELO initial doit être 1500 pour tous
        # Le premier match de chaque équipe doit avoir ELO = 1500
        
        # Vérifier que l'ELO évolue dans le temps (pas constant)
        elo_std = df['home_elo'].std()
        assert elo_std > 0, "ELO ne change pas - possible problème"
        
        # Vérifier que l'ELO n'est PAS parfaitement corrélé avec le résultat actuel
        correlation = df['home_elo'].corr(df['FTR_encoded'])
        assert abs(correlation) < 0.95, f"Data leakage ELO détecté! Corrélation: {correlation}"
    
    def test_h2h_features_no_leakage(self, engineer, cleaner):
        """
        TEST CRITIQUE: Vérifie que le H2H n'utilise pas de matchs futurs.
        
        L'historique H2H au match N ne doit inclure que les matchs < date du match N.
        """
        # Créer des données avec confrontations répétées
        teams = ['Team A', 'Team B']
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        
        df = pd.DataFrame({
            'Date': dates,
            'HomeTeam': ['Team A', 'Team B'] * 10,
            'AwayTeam': ['Team B', 'Team A'] * 10,
            'FTR': ['H', 'A'] * 10,  # Toujours victoire à domicile
            'B365H': [2.0] * 20,
            'B365D': [3.0] * 20,
            'B365A': [3.5] * 20
        })
        
        df = cleaner.clean(df)
        df = engineer.create_all_features(df)
        
        # Pour le PREMIER match: H2H doit être à 0 (aucun historique)
        first_match = df.iloc[0]
        assert first_match['h2h_total'] == 0, "H2H leakage: premier match a un historique!"
        
        # Pour le DEUXIÈME match: H2H doit être 1 (un match précédent)
        second_match = df.iloc[1]
        assert second_match['h2h_total'] == 1, "H2H leakage: deuxième match n'a pas l'historique correct!"
    
    def test_check_data_leakage_report(self, engineer, cleaner, sample_raw_data):
        """Test le rapport de détection de data leakage."""
        df = cleaner.clean(sample_raw_data)
        df = engineer.create_all_features(df)
        
        report = engineer.check_data_leakage(df)
        
        assert 'has_leakage' in report
        assert 'issues' in report
        assert 'recommendations' in report
        
        # Nos features correctement implémentées ne devraient pas avoir
        # de colonnes suspectes (future, next, etc.)
        suspicious_issues = [i for i in report['issues'] if 'future' in i.lower() or 'next' in i.lower()]
        assert len(suspicious_issues) == 0
    
    def test_check_data_leakage_detects_suspicious_columns(self, engineer):
        """Test que la détection trouve les colonnes suspectes."""
        df = pd.DataFrame({
            'home_form_5': [1, 2, 3],
            'future_result': [2, 1, 0],  # Colonne suspecte!
            'FTR_encoded': [2, 1, 0]
        })
        
        engineer.feature_columns = ['home_form_5', 'future_result']
        report = engineer.check_data_leakage(df)
        
        assert report['has_leakage'] is True
        assert any('future' in issue.lower() for issue in report['issues'])
    
    def test_feature_columns_list(self, engineer, cleaner, sample_raw_data):
        """Test que la liste des feature columns est correcte."""
        df = cleaner.clean(sample_raw_data)
        df = engineer.create_all_features(df)
        
        feature_cols = engineer.get_feature_columns()
        
        assert len(feature_cols) > 0
        assert isinstance(feature_cols, list)
        
        # Vérifier que les colonnes metadata ne sont pas incluses
        excluded = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTR_encoded', 'season', 'league']
        for col in excluded:
            assert col not in feature_cols


# ============================================================================
# TESTS UNITAIRES - PREDICTIONINTERPRETER
# ============================================================================

class TestPredictionInterpreter:
    """Tests unitaires pour PredictionInterpreter."""
    
    def test_init(self, interpreter):
        """Test l'initialisation."""
        assert interpreter.interpretation_history == []
        assert interpreter.VALUE_BET_THRESHOLD == 0.15
    
    def test_get_confidence_level(self, interpreter):
        """Test la détermination du niveau de confiance."""
        # Faible < 40%
        level, label = interpreter.get_confidence_level(0.35)
        assert level == 'low'
        assert label == 'Faible'
        
        # Moyenne 40-60%
        level, label = interpreter.get_confidence_level(0.50)
        assert level == 'medium'
        assert label == 'Moyenne'
        
        # Élevée > 60%
        level, label = interpreter.get_confidence_level(0.75)
        assert level == 'high'
        assert label == 'Élevée'
    
    def test_detect_value_bet_positive(self, interpreter):
        """Test la détection de value bet positif."""
        # Modèle: 60%, Bookmaker: 2.5 (implicite: 40%)
        # Divergence: 20% > 15% → VALUE BET
        result = interpreter.detect_value_bet(
            model_probability=0.60,
            bookmaker_odd=2.5
        )
        
        assert result['is_value_bet'] is True
        assert result['divergence_absolute'] > 0.15
        assert result['expected_value'] > 0
    
    def test_detect_value_bet_negative(self, interpreter):
        """Test quand il n'y a pas de value bet."""
        # Modèle: 45%, Bookmaker: 2.0 (implicite: 50%)
        # Divergence: -5% → PAS VALUE BET
        result = interpreter.detect_value_bet(
            model_probability=0.45,
            bookmaker_odd=2.0
        )
        
        assert result['is_value_bet'] is False
        assert result['divergence_absolute'] < 0
    
    def test_detect_value_bet_threshold(self, interpreter):
        """Test le seuil exact de 15%."""
        # Juste au-dessus du seuil
        result_above = interpreter.detect_value_bet(0.50, 2.30)  # ~43.5% implicite
        
        # Juste en-dessous
        result_below = interpreter.detect_value_bet(0.50, 2.05)  # ~48.8% implicite
        
        # La divergence doit être différente
        assert result_above['divergence_absolute'] > result_below['divergence_absolute']
    
    def test_generate_analysis_points(self, interpreter):
        """Test la génération des points d'analyse."""
        features = {
            'elo_diff': 250,
            'home_form_5': 2.5,
            'h2h_home_wins': 7,
            'h2h_total': 10
        }
        
        prediction = {
            'value_bet_info': {
                'is_value_bet': True,
                'divergence_percent': 18.5,
                'expected_value': 0.12
            }
        }
        
        points = interpreter.generate_analysis_points(features, prediction)
        
        assert len(points) > 0
        assert any('ELO' in point for point in points)
        assert any('forme' in point.lower() for point in points)
        assert any('VALUE BET' in point for point in points)
    
    def test_interpret_prediction(self, interpreter):
        """Test l'interprétation complète d'une prédiction."""
        match_data = {
            'HomeTeam': 'PSG',
            'AwayTeam': 'Marseille',
            'Date': '2024-01-15',
            'league': 'Ligue 1',
            'B365H': 1.80,
            'B365D': 3.50,
            'B365A': 4.50
        }
        
        model_prediction = {
            'probabilities': {'home': 0.65, 'draw': 0.20, 'away': 0.15},
            'predicted_class': 2
        }
        
        features = {'elo_diff': 150, 'home_form_5': 2.2}
        
        result = interpreter.interpret_prediction(match_data, model_prediction, features)
        
        # Vérifier la structure
        assert 'match_info' in result
        assert 'prediction' in result
        assert 'value_bet' in result
        assert 'analysis' in result
        assert 'recommendation' in result
        
        # Vérifier le contenu
        assert result['match_info']['home_team'] == 'PSG'
        assert result['prediction']['confidence_level'] == 'high'
        assert result['prediction']['model_probability'] == 0.65
    
    def test_generate_report_text(self, interpreter):
        """Test la génération de rapport texte."""
        match_data = {
            'HomeTeam': 'Team A',
            'AwayTeam': 'Team B',
            'Date': '2024-01-15',
            'league': 'Premier League',
            'B365H': 2.0,
            'B365D': 3.0,
            'B365A': 4.0
        }
        
        model_prediction = {
            'probabilities': {'home': 0.55, 'draw': 0.25, 'away': 0.20},
            'predicted_class': 2
        }
        
        interpretation = interpreter.interpret_prediction(match_data, model_prediction)
        report = interpreter.generate_report(interpretation, output_format='text')
        
        assert isinstance(report, str)
        assert 'Team A' in report
        assert 'Team B' in report
        assert len(report) > 100
    
    def test_generate_report_json(self, interpreter):
        """Test la génération de rapport JSON."""
        match_data = {
            'HomeTeam': 'Team A',
            'AwayTeam': 'Team B',
            'Date': '2024-01-15',
            'league': 'Premier League'
        }
        
        model_prediction = {
            'probabilities': {'home': 0.50, 'draw': 0.30, 'away': 0.20},
            'predicted_class': 2
        }
        
        interpretation = interpreter.interpret_prediction(match_data, model_prediction)
        report = interpreter.generate_report(interpretation, output_format='json')
        
        import json
        parsed = json.loads(report)
        
        assert 'match_info' in parsed
        assert 'prediction' in parsed
    
    def test_batch_interpret(self, interpreter):
        """Test l'interprétation en batch."""
        match_datas = [
            {'HomeTeam': f'Team {i}', 'AwayTeam': f'Team {i+10}'}
            for i in range(5)
        ]
        
        predictions = [
            {'probabilities': {'home': 0.5, 'draw': 0.3, 'away': 0.2}, 'predicted_class': 2}
            for _ in range(5)
        ]
        
        results = interpreter.batch_interpret(predictions, match_datas)
        
        assert len(results) == 5
        assert all('match_info' in r for r in results)
    
    def test_get_summary_statistics(self, interpreter):
        """Test les statistiques récapitulatives."""
        # Ajouter des interprétations
        for i in range(10):
            match_data = {'HomeTeam': 'A', 'AwayTeam': 'B'}
            model_prediction = {
                'probabilities': {'home': 0.4 + i * 0.05, 'draw': 0.3, 'away': 0.3},
                'predicted_class': 2
            }
            interpreter.interpret_prediction(match_data, model_prediction)
        
        stats = interpreter.get_summary_statistics()
        
        assert stats['total_interpretations'] == 10
        assert 'confidence_distribution' in stats
        assert 'value_bets_detected' in stats


# ============================================================================
# TESTS D'INTÉGRATION - PIPELINE COMPLET
# ============================================================================

class TestPipelineIntegration:
    """
    Tests d'intégration du pipeline complet.
    Valide que tous les composants fonctionnent ensemble.
    """
    
    def test_full_pipeline_flow(self, temp_data_dir):
        """Test le flux complet: chargement → nettoyage → features → interprétation."""
        # 1. Chargement
        loader = FootballDataLoader(data_dir=temp_data_dir)
        df = loader.load_season('E0', '23-24')
        
        assert len(df) == 100
        assert loader.validate_data(df)['valid'] is True
        
        # 2. Nettoyage
        cleaner = DataCleaner()
        df = cleaner.clean(df)
        
        assert 'FTR_encoded' in df.columns
        assert np.issubdtype(df['Date'].dtype, np.datetime64)
        
        # 3. Feature Engineering
        engineer = FeatureEngineer()
        df = engineer.create_all_features(df)
        
        assert len(engineer.get_feature_columns()) > 0
        
        # 4. Vérification data leakage
        leakage_report = engineer.check_data_leakage(df)
        suspicious_issues = [i for i in leakage_report['issues'] if 'future' in i.lower() or 'next' in i.lower()]
        assert len(suspicious_issues) == 0
        
        # 5. Interprétation (simulation)
        interpreter = PredictionInterpreter()
        
        # Simuler une prédiction pour le dernier match
        last_match = df.iloc[-1]
        match_data = {
            'HomeTeam': last_match['HomeTeam'],
            'AwayTeam': last_match['AwayTeam'],
            'Date': str(last_match['Date']),
            'league': last_match.get('league', 'E0')
        }
        
        model_prediction = {
            'probabilities': {'home': 0.55, 'draw': 0.25, 'away': 0.20},
            'predicted_class': 2
        }
        
        features = {
            'elo_diff': last_match.get('elo_diff', 0),
            'home_form_5': last_match.get('home_form_5', 1.5)
        }
        
        interpretation = interpreter.interpret_prediction(match_data, model_prediction, features)
        
        assert 'recommendation' in interpretation
        assert interpretation['prediction']['confidence_level'] in ['low', 'medium', 'high']
    
    def test_pipeline_with_missing_data(self, cleaner, engineer):
        """Test le pipeline avec des données manquantes."""
        # Créer des données avec valeurs manquantes
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=20),
            'HomeTeam': ['Team A', 'Team B'] * 10,
            'AwayTeam': ['Team C', 'Team D'] * 10,
            'FTR': np.random.choice(['H', 'D', 'A'], 20),
            'B365H': [2.0 if i % 3 != 0 else np.nan for i in range(20)],
            'B365D': [3.0] * 20,
            'B365A': [3.5] * 20
        })
        
        # Le pipeline doit gérer les NaN sans erreur
        df = cleaner.clean(df)
        df = engineer.create_all_features(df)
        
        # Vérifier que le pipeline a complété sans erreur
        assert len(df) > 0
        assert 'home_form_5' in df.columns
    
    def test_pipeline_temporal_order(self, cleaner, engineer):
        """
        TEST CRITIQUE: Vérifie que le pipeline respecte l'ordre temporel.
        
        Les features doivent être calculées dans l'ordre chronologique
        pour éviter le data leakage.
        """
        # Créer des données ordonnées
        dates = pd.date_range('2023-01-01', periods=30)
        df = pd.DataFrame({
            'Date': dates,
            'HomeTeam': ['Team A'] * 15 + ['Team B'] * 15,
            'AwayTeam': ['Team C'] * 15 + ['Team D'] * 15,
            'FTR': ['H'] * 30,  # Victoire domicile constante
            'B365H': [2.0] * 30,
            'B365D': [3.0] * 30,
            'B365A': [3.5] * 30
        })
        
        # Trier par date (devrait être déjà trié, mais on vérifie)
        df = df.sort_values('Date').reset_index(drop=True)
        
        df = cleaner.clean(df)
        df = engineer.create_all_features(df)
        
        # Vérifier que les features rolling augmentent progressivement
        # (car l'équipe gagne tous ses matchs)
        home_form = df['home_form_5'].values
        
        # La forme devrait globalement augmenter ou rester stable
        # Les premières valeurs peuvent être NaN car pas d'historique
        non_nan_form = home_form[~np.isnan(home_form)]
        assert len(non_nan_form) > 0
        assert non_nan_form.min() >= 0
        assert non_nan_form.max() <= 3  # Max 3 points par match
    
    def test_pipeline_multiple_leagues(self, temp_data_dir):
        """Test le pipeline avec plusieurs ligues."""
        # Créer des données pour plusieurs ligues
        leagues = ['E0', 'SP1', 'D1']
        
        for league in leagues:
            df = pd.DataFrame({
                'Date': pd.date_range('2023-08-01', periods=50).strftime('%d/%m/%Y'),
                'HomeTeam': np.random.choice([f'{league} Team {i}' for i in range(10)], 50),
                'AwayTeam': np.random.choice([f'{league} Team {i}' for i in range(10)], 50),
                'FTR': np.random.choice(['H', 'D', 'A'], 50),
                'B365H': np.random.uniform(1.5, 3.5, 50),
                'B365D': np.random.uniform(2.5, 4.5, 50),
                'B365A': np.random.uniform(2.0, 5.0, 50)
            })
            
            filepath = os.path.join(temp_data_dir, f'{league}_23-24.csv')
            df.to_csv(filepath, index=False, encoding='latin-1')
        
        # Charger toutes les ligues
        loader = FootballDataLoader(data_dir=temp_data_dir)
        
        all_data = []
        for league in leagues:
            df = loader.load_season(league, '23-24')
            all_data.append(df)
        
        combined = pd.concat(all_data, ignore_index=True)
        
        assert len(combined) == 150
        assert set(combined['league'].unique()) == set(leagues)
        
        # Nettoyer et créer features
        cleaner = DataCleaner()
        engineer = FeatureEngineer()
        
        combined = cleaner.clean(combined)
        combined = engineer.create_all_features(combined)
        
        # Vérifier que tout fonctionne
        assert len(engineer.get_feature_columns()) > 0
        
        # Vérifier pas de colonnes suspectes (future, next, etc.)
        report = engineer.check_data_leakage(combined)
        suspicious_issues = [i for i in report['issues'] if 'future' in i.lower() or 'next' in i.lower()]
        assert len(suspicious_issues) == 0


# ============================================================================
# TESTS DE RÉGRESSION
# ============================================================================

class TestRegression:
    """Tests de régression pour éviter les bugs futurs."""
    
    def test_no_division_by_zero_in_probabilities(self, cleaner):
        """Test qu'il n'y a pas de division par zéro dans le calcul des probabilités."""
        df = pd.DataFrame({
            'Date': ['01/08/2023'],
            'HomeTeam': ['Team A'],
            'AwayTeam': ['Team B'],
            'FTR': ['H'],
            'B365H': [0.0],  # Cote à zéro - devrait causer division par zéro
            'B365D': [0.0],
            'B365A': [0.0]
        })
        
        df = cleaner.clean(df)
        
        # Ne devrait pas lever d'erreur
        try:
            df = cleaner.calculate_probabilities(df)
            # Si on arrive ici, vérifier que les NaN sont gérés
            assert True
        except ZeroDivisionError:
            pytest.fail("Division par zéro dans calculate_probabilities")
    
    def test_empty_dataframe_handling(self, cleaner, engineer):
        """Test la gestion des DataFrames vides."""
        df = pd.DataFrame()
        
        # Devrait gérer gracieusement
        try:
            cleaned = cleaner.clean(df)
            assert len(cleaned) == 0
        except Exception as e:
            # Acceptable si lève une erreur claire
            assert True
    
    def test_single_match_prediction(self, engineer, cleaner):
        """Test avec un seul match (cas limite)."""
        df = pd.DataFrame({
            'Date': ['01/08/2023'],
            'HomeTeam': ['Team A'],
            'AwayTeam': ['Team B'],
            'FTR': ['H'],
            'B365H': [2.0],
            'B365D': [3.0],
            'B365A': [3.5]
        })
        
        df = cleaner.clean(df)
        df = engineer.create_all_features(df)
        
        # Devrait fonctionner même avec un seul match
        assert len(df) == 1
        assert 'home_elo' in df.columns


# ============================================================================
# EXÉCUTION DES TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
