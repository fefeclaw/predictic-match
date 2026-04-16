"""
Module de feature engineering pour la prédiction football
Respect strict : aucune donnée du futur (data leakage prevention)
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass


# =============================================================================
# CLASSE FeatureEngineer - Rolling averages, stats domicile/extérieur, forme
# =============================================================================

class FeatureEngineer:
    """
    Crée des features temporelles pour chaque équipe.
    Toutes les features sont calculées UNIQUEMENT sur les matchs PASSÉS.
    """
    
    def __init__(self, rolling_windows: List[int] = None):
        """
        Initialise le FeatureEngineer.
        
        Args:
            rolling_windows: Fenêtres pour les moyennes mobiles (défaut: [5, 10, 20])
        """
        self.rolling_windows = rolling_windows or [5, 10, 20]
        self.feature_columns: List[str] = []
        
    def create_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée toutes les features d'équipe en respectant l'ordre temporel.
        
        Args:
            df: DataFrame TRIÉ par date (chronologique)
            
        Returns:
            DataFrame avec features ajoutées
        """
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Stats de base
        df = self._add_basic_stats(df)
        
        # Rolling averages (forme récente)
        df = self._add_rolling_averages(df)
        
        # Stats domicile/extérieur
        df = self._add_home_away_splits(df)
        
        return df
    
    def _add_basic_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les stats de base (points, buts) par match."""
        # Points obtenus
        df['_home_points'] = df['FTR_encoded'].map({2: 3, 1: 1, 0: 0})
        df['_away_points'] = df['FTR_encoded'].map({2: 0, 1: 1, 0: 3})
        
        return df
    
    def _add_rolling_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les moyennes mobiles sur les N derniers matchs.
        CRITIQUE: Utilise shift(1) pour éviter le leakage.
        """
        for window in self.rolling_windows:
            # Forme générale (points par match)
            df[f'home_form_{window}'] = df.groupby('HomeTeam').apply(
                lambda x: x['_home_points'].shift(1).rolling(window=window, min_periods=1).mean()
            ).reset_index(level=0, drop=True)
            
            df[f'away_form_{window}'] = df.groupby('AwayTeam').apply(
                lambda x: x['_away_points'].shift(1).rolling(window=window, min_periods=1).mean()
            ).reset_index(level=0, drop=True)
            
            # Buts marqués
            if 'FTHG' in df.columns:
                df[f'home_goals_scored_{window}'] = df.groupby('HomeTeam').apply(
                    lambda x: x['FTHG'].shift(1).rolling(window=window, min_periods=1).mean()
                ).reset_index(level=0, drop=True)
                
            if 'FTAG' in df.columns:
                df[f'away_goals_scored_{window}'] = df.groupby('AwayTeam').apply(
                    lambda x: x['FTAG'].shift(1).rolling(window=window, min_periods=1).mean()
                ).reset_index(level=0, drop=True)
            
            # Buts encaissés (inversé : buts encaissés à domicile = buts marqués par l'adversaire)
            if 'FTAG' in df.columns:
                df[f'home_goals_conceded_{window}'] = df.groupby('HomeTeam').apply(
                    lambda x: x['FTAG'].shift(1).rolling(window=window, min_periods=1).mean()
                ).reset_index(level=0, drop=True)
                
            if 'FTHG' in df.columns:
                df[f'away_goals_conceded_{window}'] = df.groupby('AwayTeam').apply(
                    lambda x: x['FTHG'].shift(1).rolling(window=window, min_periods=1).mean()
                ).reset_index(level=0, drop=True)
            
            # Différence de buts
            if f'home_goals_scored_{window}' in df.columns and f'home_goals_conceded_{window}' in df.columns:
                df[f'home_gd_{window}'] = df[f'home_goals_scored_{window}'] - df[f'home_goals_conceded_{window}']
                df[f'away_gd_{window}'] = df[f'away_goals_scored_{window}'] - df[f'away_goals_conceded_{window}']
        
        return df
    
    def _add_home_away_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stats spécifiques domicile vs extérieur.
        Une équipe peut performer différemment à domicile et à l'extérieur.
        """
        window = 5  # Forme récente sur 5 matchs
        
        # Performance à domicile pour l'équipe home (quand elle joue à domicile)
        df['home_home_form'] = df.groupby('HomeTeam').apply(
            lambda x: x['_home_points'].shift(1).rolling(window=window, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
        
        # Performance à l'extérieur pour l'équipe away (quand elle joue à l'extérieur)
        df['away_away_form'] = df.groupby('AwayTeam').apply(
            lambda x: x['_away_points'].shift(1).rolling(window=window, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
        
        # Performance inverse (équipe home quand elle joue à l'extérieur, etc.)
        # Nécessite de tracker les matchs où l'équipe était away
        df = self._add_cross_venue_stats(df, window)
        
        return df
    
    def _add_cross_venue_stats(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Ajoute les stats croisées (performance à l'extérieur pour équipe home, etc.)
        """
        # Créer un DataFrame unifié pour tracker toutes les performances
        home_matches = df[['Date', 'HomeTeam', '_home_points', 'FTHG', 'FTAG']].copy()
        home_matches.columns = ['Date', 'Team', 'points', 'goals_for', 'goals_against']
        home_matches['venue'] = 'home'
        
        away_matches = df[['Date', 'AwayTeam', '_away_points', 'FTAG', 'FTHG']].copy()
        away_matches.columns = ['Date', 'Team', 'points', 'goals_for', 'goals_against']
        away_matches['venue'] = 'away'
        
        all_matches = pd.concat([home_matches, away_matches], ignore_index=True)
        all_matches = all_matches.sort_values(['Team', 'Date'])
        
        # Stats à domicile (pour toutes les équipes)
        home_only = all_matches[all_matches['venue'] == 'home']
        df['team_home_form'] = df.apply(
            lambda row: home_only[home_only['Team'] == row['HomeTeam']]['points'].shift(1).rolling(window=window, min_periods=1).mean().iloc[-1] if len(home_only[home_only['Team'] == row['HomeTeam']]) > 0 else np.nan,
            axis=1
        )
        
        # Stats à l'extérieur
        away_only = all_matches[all_matches['venue'] == 'away']
        df['team_away_form'] = df.apply(
            lambda row: away_only[away_only['Team'] == row['AwayTeam']]['points'].shift(1).rolling(window=window, min_periods=1).mean().iloc[-1] if len(away_only[away_only['Team'] == row['AwayTeam']]) > 0 else np.nan,
            axis=1
        )
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Retourne la liste des colonnes features."""
        return self.feature_columns


# =============================================================================
# CLASSE FootballELO - Système ELO avec margin of victory (formule FIFA)
# =============================================================================

class FootballELO:
    """
    Système de rating ELO adapté au football avec margin of victory.
    Basé sur la formule FIFA/ELO avec ajustements pour le football.
    """
    
    def __init__(self, k_factor: int = 32, home_advantage: int = 50):
        """
        Initialise le système ELO.
        
        Args:
            k_factor: Facteur K pour les mises à jour (défaut: 32)
            home_advantage: Bonus ELO pour l'équipe à domicile (défaut: 50)
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.elo_ratings: Dict[str, float] = {}
        self.elo_history: Dict[str, List[Tuple[int, float]]] = {}
        
    def get_elo_before_match(self, home_team: str, away_team: str, match_date: datetime) -> Tuple[float, float]:
        """
        Récupère les ratings ELO AVANT un match (pour éviter leakage).
        
        Args:
            home_team: Nom de l'équipe à domicile
            away_team: Nom de l'équipe à l'extérieur
            match_date: Date du match
            
        Returns:
            Tuple (home_elo, away_elo) avant le match
        """
        home_elo = self.elo_ratings.get(home_team, 1500)
        away_elo = self.elo_ratings.get(away_team, 1500)
        
        # Appliquer avantage domicile
        home_elo_with_advantage = home_elo + self.home_advantage
        
        return home_elo_with_advantage, away_elo
    
    def update_elo(self, home_team: str, away_team: str, 
                   home_goals: int, away_goals: int, 
                   match_date: datetime) -> None:
        """
        Met à jour les ratings ELO APRÈS un match.
        
        Args:
            home_team: Équipe à domicile
            away_team: Équipe à l'extérieur
            home_goals: Buts marqués par home
            away_goals: Buts marqués par away
            match_date: Date du match
        """
        # Initialiser si nécessaire
        for team in [home_team, away_team]:
            if team not in self.elo_ratings:
                self.elo_ratings[team] = 1500
                self.elo_history[team] = []
        
        # ELO avant match (sans avantage domicile pour le calcul)
        home_elo = self.elo_ratings[home_team]
        away_elo = self.elo_ratings[away_team]
        
        # Score attendu (avec avantage domicile)
        elo_diff = (home_elo + self.home_advantage) - away_elo
        expected_home = 1 / (1 + 10 ** (-elo_diff / 400))
        expected_away = 1 - expected_home
        
        # Score réel
        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals < away_goals:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5
        
        # Margin of Victory multiplier (formule FIFA)
        goal_diff = abs(home_goals - away_goals)
        if goal_diff == 0:
            mov_multiplier = 1.0
        elif goal_diff == 1:
            mov_multiplier = 1.0
        elif goal_diff == 2:
            mov_multiplier = 1.25
        elif goal_diff == 3:
            mov_multiplier = 1.5
        else:
            mov_multiplier = 1.5 + (goal_diff - 3) * 0.1
        mov_multiplier = min(mov_multiplier, 2.0)  # Cap à 2.0
        
        # Bonus pour match important (optionnel : détecté via contexte)
        importance_multiplier = 1.0
        
        # Mise à jour ELO
        delta_home = self.k_factor * mov_multiplier * importance_multiplier * (actual_home - expected_home)
        delta_away = self.k_factor * mov_multiplier * importance_multiplier * (actual_away - expected_away)
        
        self.elo_ratings[home_team] += delta_home
        self.elo_ratings[away_team] += delta_away
        
        # Historique
        self.elo_history[home_team].append((match_date, self.elo_ratings[home_team]))
        self.elo_history[away_team].append((match_date, self.elo_ratings[away_team]))
    
    def compute_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les features ELO pour tout le DataFrame.
        CRITIQUE: Utilise uniquement les ratings AVANT chaque match.
        
        Args:
            df: DataFrame TRIÉ par date
            
        Returns:
            DataFrame avec features ELO ajoutées
        """
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Réinitialiser les ratings
        self.elo_ratings = {}
        self.elo_history = {}
        
        home_elo_list = []
        away_elo_list = []
        elo_diff_list = []
        
        for idx, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            match_date = row['Date']
            
            # ELO AVANT le match (pas de leakage!)
            home_elo, away_elo = self.get_elo_before_match(home_team, away_team, match_date)
            
            home_elo_list.append(home_elo)
            away_elo_list.append(away_elo)
            elo_diff_list.append(home_elo - away_elo)
            
            # Mettre à jour APRÈS avec le résultat
            if 'FTHG' in row and 'FTAG' in row:
                self.update_elo(home_team, away_team, row['FTHG'], row['FTAG'], match_date)
            elif 'FTR_encoded' in row:
                # Si pas de score exact, utiliser une estimation
                result = row['FTR_encoded']
                if result == 2:  # Home win
                    self.update_elo(home_team, away_team, 2, 0, match_date)
                elif result == 0:  # Away win
                    self.update_elo(home_team, away_team, 0, 2, match_date)
                else:  # Draw
                    self.update_elo(home_team, away_team, 1, 1, match_date)
        
        df['home_elo'] = home_elo_list
        df['away_elo'] = away_elo_list
        df['elo_diff'] = elo_diff_list
        
        return df
    
    def get_current_ratings(self) -> Dict[str, float]:
        """Retourne les ratings ELO actuels."""
        return self.elo_ratings.copy()


# =============================================================================
# FONCTION compute_xg_proxy - Expected Goals proxy
# =============================================================================

def compute_xg_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule un proxy xG (expected goals) basé sur les tirs et opportunités.
    
    En l'absence de données xG réelles, utilise :
    - Tirs (Shots)
    - Tirs cadrés (Shots on Target)
    - Corners
    - Fautes commises (proxy pour la pression)
    
    Args:
        df: DataFrame avec stats de match
        
    Returns:
        DataFrame avec xG proxy ajouté
    """
    # Si xG existe déjà, ne pas recalculer
    if 'xG' in df.columns and 'xGA' in df.columns:
        return df
    
    # Proxy basé sur les tirs
    if 'HS' in df.columns and 'AS' in df.columns:  # Home/Away Shots
        # Formule simplifiée : xG ≈ tirs * 0.1 (moyenne football)
        df['home_xg_proxy'] = df['HS'] * 0.10
        df['away_xg_proxy'] = df['AS'] * 0.10
        
        # Ajustement si tirs cadrés disponibles
        if 'HST' in df.columns and 'AST' in df.columns:
            # Tirs cadrés ont plus de poids
            df['home_xg_proxy'] = df['HS'] * 0.07 + df['HST'] * 0.15
            df['away_xg_proxy'] = df['AS'] * 0.07 + df['AST'] * 0.15
    
    # Proxy basé sur les corners (opportunités créées)
    if 'HC' in df.columns and 'AC' in df.columns:
        df['home_xg_from_corners'] = df['HC'] * 0.03  # ~3% de xG par corner
        df['away_xg_from_corners'] = df['AC'] * 0.03
        
        if 'home_xg_proxy' in df.columns:
            df['home_xg_proxy'] += df['home_xg_from_corners']
            df['away_xg_proxy'] += df['away_xg_from_corners']
        else:
            df['home_xg_proxy'] = df['home_xg_from_corners']
            df['away_xg_proxy'] = df['away_xg_from_corners']
    
    # Rolling xG (forme offensive)
    for window in [5, 10]:
        if 'home_xg_proxy' in df.columns:
            df[f'home_xg_rolling_{window}'] = df.groupby('HomeTeam').apply(
                lambda x: x['home_xg_proxy'].shift(1).rolling(window=window, min_periods=1).mean()
            ).reset_index(level=0, drop=True)
            
            df[f'away_xg_rolling_{window}'] = df.groupby('AwayTeam').apply(
                lambda x: x['away_xg_proxy'].shift(1).rolling(window=window, min_periods=1).mean()
            ).reset_index(level=0, drop=True)
    
    return df


# =============================================================================
# FONCTION compute_fatigue_features - Facteurs de fatigue
# =============================================================================

def compute_fatigue_features(df: pd.DataFrame, lookback_days: int = 14) -> pd.DataFrame:
    """
    Calcule les features de fatigue basées sur :
    - Jours de repos depuis le dernier match
    - Nombre de matchs dans les X derniers jours
    - Voyages (si données de localisation disponibles)
    
    Args:
        df: DataFrame avec dates de match
        lookback_days: Fenêtre pour calculer la densité de matchs
        
    Returns:
        DataFrame avec features de fatigue
    """
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Jours depuis le dernier match
    df['home_days_rest'] = df.groupby('HomeTeam')['Date'].diff().dt.days
    df['away_days_rest'] = df.groupby('AwayTeam')['Date'].diff().dt.days
    
    # Remplir les NaN (premier match de la saison) avec une valeur par défaut
    df['home_days_rest'] = df['home_days_rest'].fillna(7)  # 7 jours par défaut
    df['away_days_rest'] = df['away_days_rest'].fillna(7)
    
    # Nombre de matchs dans les X derniers jours
    def count_recent_matches(group: pd.DataFrame, current_date: datetime, days: int) -> int:
        """Compte les matchs dans les X jours avant la date actuelle."""
        recent = group[
            (group['Date'] < current_date) & 
            (group['Date'] >= current_date - timedelta(days=days))
        ]
        return len(recent)
    
    # Pour chaque équipe, compter les matchs récents
    home_matches_count = []
    away_matches_count = []
    
    for idx, row in df.iterrows():
        # Équipe home
        home_history = df[
            (df['HomeTeam'] == row['HomeTeam']) | (df['AwayTeam'] == row['HomeTeam'])
        ]
        home_count = count_recent_matches(home_history, row['Date'], lookback_days)
        home_matches_count.append(home_count)
        
        # Équipe away
        away_history = df[
            (df['HomeTeam'] == row['AwayTeam']) | (df['AwayTeam'] == row['AwayTeam'])
        ]
        away_count = count_recent_matches(away_history, row['Date'], lookback_days)
        away_matches_count.append(away_count)
    
    df['home_recent_matches'] = home_matches_count
    df['away_recent_matches'] = away_matches_count
    
    # Feature composite de fatigue
    # Moins de repos + plus de matchs = plus de fatigue
    df['home_fatigue_score'] = (
        (14 - df['home_days_rest'].clip(0, 14)) / 14 * 0.5 +  # 0-0.5
        (df['home_recent_matches'] / 5).clip(0, 1) * 0.5  # 0-0.5
    )
    
    df['away_fatigue_score'] = (
        (14 - df['away_days_rest'].clip(0, 14)) / 14 * 0.5 +
        (df['away_recent_matches'] / 5).clip(0, 1) * 0.5
    )
    
    # Différence de fatigue (avantage pour l'équipe moins fatiguée)
    df['fatigue_diff'] = df['away_fatigue_score'] - df['home_fatigue_score']
    
    return df


# =============================================================================
# FONCTION compute_h2h_features - Head-to-Head
# =============================================================================

def compute_h2h_features(df: pd.DataFrame, max_history: int = 10) -> pd.DataFrame:
    """
    Calcule les features Head-to-Head entre les deux équipes.
    CRITIQUE: Utilise UNIQUEMENT les matchs PASSÉS avant la date du match.
    
    Args:
        df: DataFrame TRIÉ par date
        max_history: Nombre maximum de confrontations à considérer
        
    Returns:
        DataFrame avec features H2H
    """
    df = df.sort_values('Date').reset_index(drop=True)
    
    h2h_home_wins = []
    h2h_away_wins = []
    h2h_draws = []
    h2h_home_goals = []
    h2h_away_goals = []
    h2h_total_goals = []
    
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        current_date = row['Date']
        
        # Matchs PASSÉS entre ces équipes (strictement avant current_date)
        mask = (
            ((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) |
            ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))
        ) & (df['Date'] < current_date)
        
        h2h_history = df[mask].tail(max_history)
        
        # Compter les résultats
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals_sum = 0
        away_goals_sum = 0
        
        for _, h2h_row in h2h_history.iterrows():
            if h2h_row['HomeTeam'] == home:
                home_goals_sum += h2h_row.get('FTHG', 0)
                away_goals_sum += h2h_row.get('FTAG', 0)
                if h2h_row['FTR'] == 'H':
                    home_wins += 1
                elif h2h_row['FTR'] == 'A':
                    away_wins += 1
                else:
                    draws += 1
            else:
                home_goals_sum += h2h_row.get('FTAG', 0)
                away_goals_sum += h2h_row.get('FTHG', 0)
                if h2h_row['FTR'] == 'A':
                    home_wins += 1
                elif h2h_row['FTR'] == 'H':
                    away_wins += 1
                else:
                    draws += 1
        
        h2h_home_wins.append(home_wins)
        h2h_away_wins.append(away_wins)
        h2h_draws.append(draws)
        h2h_home_goals.append(home_goals_sum)
        h2h_away_goals.append(away_goals_sum)
        h2h_total_goals.append(home_goals_sum + away_goals_sum)
    
    df['h2h_home_wins'] = h2h_home_wins
    df['h2h_away_wins'] = h2h_away_wins
    df['h2h_draws'] = h2h_draws
    df['h2h_home_goals'] = h2h_home_goals
    df['h2h_away_goals'] = h2h_away_goals
    df['h2h_total_goals'] = h2h_total_goals
    df['h2h_total'] = df['h2h_home_wins'] + df['h2h_away_wins'] + df['h2h_draws']
    
    # Ratios H2H
    df['h2h_home_win_rate'] = df['h2h_home_wins'] / df['h2h_total'].replace(0, 1)
    df['h2h_away_win_rate'] = df['h2h_away_wins'] / df['h2h_total'].replace(0, 1)
    df['h2h_avg_goals'] = df['h2h_total_goals'] / df['h2h_total'].replace(0, 1)
    
    return df


# =============================================================================
# FONCTION add_odds_features - Probabilités bookmaker normalisées
# =============================================================================

def add_odds_features(df: pd.DataFrame, odds_cols: Dict[str, str] = None) -> pd.DataFrame:
    """
    Ajoute les features dérivées des cotes bookmaker.
    
    Args:
        df: DataFrame avec cotes
        odds_cols: Mapping des colonnes de cotes 
                   {'home': 'B365H', 'draw': 'B365D', 'away': 'B365A'}
                   
    Returns:
        DataFrame avec features de cotes
    """
    if odds_cols is None:
        odds_cols = {'home': 'B365H', 'draw': 'B365D', 'away': 'B365A'}
    
    # Vérifier si les colonnes existent
    if not all(col in df.columns for col in odds_cols.values()):
        # Essayer d'autres bookmakers
        alternative_bookmakers = ['Bet365', 'B365', 'IW', 'WH', 'VC']
        for bookie in alternative_bookmakers:
            alt_cols = {
                'home': f'{bookie}H',
                'draw': f'{bookie}D',
                'away': f'{bookie}A'
            }
            if all(col in df.columns for col in alt_cols.values()):
                odds_cols = alt_cols
                break
        else:
            # Pas de cotes disponibles
            return df
    
    home_col = odds_cols['home']
    draw_col = odds_cols['draw']
    away_col = odds_cols['away']
    
    # Nettoyer les cotes (remplacer les 0 ou NaN)
    for col in [home_col, draw_col, away_col]:
        df[col] = df[col].replace(0, np.nan)
    
    # Probabilités implicites brutes
    df['_implied_home'] = 1.0 / df[home_col]
    df['_implied_draw'] = 1.0 / df[draw_col]
    df['_implied_away'] = 1.0 / df[away_col]
    
    # Somme des probabilités (inclut la marge)
    df['_prob_sum'] = df['_implied_home'] + df['_implied_draw'] + df['_implied_away']
    
    # Probabilités normalisées (sans la marge)
    df['implied_prob_home'] = df['_implied_home'] / df['_prob_sum']
    df['implied_prob_draw'] = df['_implied_draw'] / df['_prob_sum']
    df['implied_prob_away'] = df['_implied_away'] / df['_prob_sum']
    
    # Marge du bookmaker (overround)
    df['bookmaker_margin'] = df['_prob_sum'] - 1
    
    # Odds ratio
    df['odds_ratio_home_away'] = df[home_col] / df[away_col]
    df['odds_ratio_home_draw'] = df[home_col] / df[draw_col]
    df['odds_ratio_draw_away'] = df[draw_col] / df[away_col]
    
    # Value bets potentiels (écart par rapport à 50/50)
    df['odds_imbalance'] = abs(df['implied_prob_home'] - df['implied_prob_away'])
    
    # Nettoyage des colonnes temporaires
    df = df.drop(columns=['_implied_home', '_implied_draw', '_implied_away', '_prob_sum'])
    
    return df


# =============================================================================
# CLASSE TripleLayerFeatures - Divergences Bookmaker vs Polymarket
# =============================================================================

@dataclass
class MarketOdds:
    """Structure pour stocker les cotes d'un marché."""
    home_prob: float
    draw_prob: float
    away_prob: float
    source: str


class TripleLayerFeatures:
    """
    Analyse les divergences entre trois couches de marché :
    1. Bookmakers traditionnels (Bet365, etc.)
    2. Polymarket (marché prédictif décentralisé)
    3. Modèle interne (prédictions ML)
    
    Les divergences peuvent signaler des value bets ou des informations asymétriques.
    """
    
    def __init__(self):
        """Initialise l'analyseur de divergences."""
        self.divergence_columns: List[str] = []
        
    def compute_divergences(self, df: pd.DataFrame, 
                           bookmaker_col: str = 'implied_prob_home',
                           polymarket_col: str = 'poly_home_prob',
                           model_col: str = 'model_home_prob') -> pd.DataFrame:
        """
        Calcule les divergences entre les trois couches de marché.
        
        Args:
            df: DataFrame avec probabilités de chaque source
            bookmaker_col: Colonne avec proba bookmaker
            polymarket_col: Colonne avec proba Polymarket
            model_col: Colonne avec proba du modèle
            
        Returns:
            DataFrame avec features de divergence
        """
        # Vérifier les colonnes disponibles
        available_cols = {
            'bookmaker': bookmaker_col if bookmaker_col in df.columns else None,
            'polymarket': polymarket_col if polymarket_col in df.columns else None,
            'model': model_col if model_col in df.columns else None
        }
        
        # Divergence Bookmaker vs Polymarket
        if available_cols['bookmaker'] and available_cols['polymarket']:
            df['div_bookie_poly'] = df[available_cols['bookmaker']] - df[available_cols['polymarket']]
            df['abs_div_bookie_poly'] = abs(df['div_bookie_poly'])
            self.divergence_columns.extend(['div_bookie_poly', 'abs_div_bookie_poly'])
            
            # Signal de value bet (si divergence > seuil)
            df['value_signal_bookie_poly'] = (df['abs_div_bookie_poly'] > 0.10).astype(int)
            self.divergence_columns.append('value_signal_bookie_poly')
        
        # Divergence Bookmaker vs Modèle
        if available_cols['bookmaker'] and available_cols['model']:
            df['div_bookie_model'] = df[available_cols['bookmaker']] - df[available_cols['model']]
            df['abs_div_bookie_model'] = abs(df['div_bookie_model'])
            self.divergence_columns.extend(['div_bookie_model', 'abs_div_bookie_model'])
            
            df['value_signal_bookie_model'] = (df['abs_div_bookie_model'] > 0.15).astype(int)
            self.divergence_columns.append('value_signal_bookie_model')
        
        # Divergence Polymarket vs Modèle
        if available_cols['polymarket'] and available_cols['model']:
            df['div_poly_model'] = df[available_cols['polymarket']] - df[available_cols['model']]
            df['abs_div_poly_model'] = abs(df['div_poly_model'])
            self.divergence_columns.extend(['div_poly_model', 'abs_div_poly_model'])
            
            df['value_signal_poly_model'] = (df['abs_div_poly_model'] > 0.15).astype(int)
            self.divergence_columns.append('value_signal_poly_model')
        
        # Consensus score (à quel point les 3 sources sont alignées)
        if all(available_cols.values()):
            df['consensus_std'] = df[[
                available_cols['bookmaker'],
                available_cols['polymarket'],
                available_cols['model']
            ]].std(axis=1)
            
            df['consensus_score'] = 1 - df['consensus_std'].clip(0, 0.5) / 0.5
            self.divergence_columns.extend(['consensus_std', 'consensus_score'])
        
        # Direction de la divergence (qui est le plus optimiste pour home?)
        if all(available_cols.values()):
            def get_most_optimistic(row):
                probs = {
                    'bookmaker': row[available_cols['bookmaker']],
                    'polymarket': row[available_cols['polymarket']],
                    'model': row[available_cols['model']]
                }
                return max(probs, key=probs.get)
            
            df['most_optimistic_source'] = df.apply(get_most_optimistic, axis=1)
            self.divergence_columns.append('most_optimistic_source')
        
        return df
    
    def compute_arbitrage_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Détecte les opportunités d'arbitrage entre les marchés.
        
        Args:
            df: DataFrame avec probabilités de chaque source
            
        Returns:
            DataFrame avec signaux d'arbitrage
        """
        # Vérifier les colonnes nécessaires
        required_probs = ['implied_prob_home', 'implied_prob_draw', 'implied_prob_away']
        poly_probs = ['poly_home_prob', 'poly_draw_prob', 'poly_away_prob']
        
        has_bookie = all(col in df.columns for col in required_probs)
        has_poly = all(col in df.columns for col in poly_probs)
        
        if has_bookie and has_poly:
            # Somme des probabilités inverses pour détecter l'arbitrage
            df['bookie_implied_sum'] = df[required_probs].sum(axis=1)
            df['poly_implied_sum'] = df[poly_probs].sum(axis=1)
            
            # Opportunité si un marché a une somme < 1 (arbitrage possible)
            df['arbitrage_bookie'] = (df['bookie_implied_sum'] < 0.95).astype(int)
            df['arbitrage_poly'] = (df['poly_implied_sum'] < 0.95).astype(int)
            
            self.divergence_columns.extend([
                'bookie_implied_sum', 'poly_implied_sum',
                'arbitrage_bookie', 'arbitrage_poly'
            ])
        
        # Cross-market arbitrage (combiner les meilleures cotes)
        if has_bookie and has_poly:
            # Prendre la probabilité maximale pour chaque outcome
            df['max_home_prob'] = df[['implied_prob_home', 'poly_home_prob']].max(axis=1)
            df['max_draw_prob'] = df[['implied_prob_draw', 'poly_draw_prob']].max(axis=1)
            df['max_away_prob'] = df[['implied_prob_away', 'poly_away_prob']].max(axis=1)
            
            df['cross_market_sum'] = (
                1/df['max_home_prob'] + 1/df['max_draw_prob'] + 1/df['max_away_prob']
            )
            df['cross_arbitrage_opportunity'] = (df['cross_market_sum'] < 1).astype(int)
            
            self.divergence_columns.extend([
                'max_home_prob', 'max_draw_prob', 'max_away_prob',
                'cross_market_sum', 'cross_arbitrage_opportunity'
            ])
        
        return df
    
    def get_divergence_columns(self) -> List[str]:
        """Retourne la liste des colonnes de divergence."""
        return self.divergence_columns


# =============================================================================
# FONCTION UTILITAIRE - Vérification data leakage
# =============================================================================

def check_data_leakage(df: pd.DataFrame, feature_cols: List[str], 
                       target_col: str = 'FTR_encoded') -> Dict:
    """
    Vérifie la présence potentielle de data leakage.
    
    Args:
        df: DataFrame avec features
        feature_cols: Liste des colonnes features
        target_col: Colonne cible
        
    Returns:
        Rapport de vérification
    """
    leakage_report = {
        "has_leakage": False,
        "issues": [],
        "warnings": [],
        "recommendations": []
    }
    
    # 1. Vérifier les colonnes suspectes (noms indiquant le futur)
    suspicious_keywords = ['future', 'next', 'tomorrow', 'after', 'post', 'result', 'final']
    suspicious_cols = [col for col in feature_cols if any(
        keyword in col.lower() for keyword in suspicious_keywords
    )]
    
    if suspicious_cols:
        leakage_report["warnings"].append(f"Colonnes aux noms suspects: {suspicious_cols}")
        leakage_report["recommendations"].append("Vérifier manuellement ces colonnes")
    
    # 2. Vérifier les corrélations trop élevées avec la target
    if target_col in df.columns:
        for col in feature_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                corr = df[col].corr(df[target_col])
                if abs(corr) > 0.85:
                    leakage_report["issues"].append(
                        f"Corrélation très élevée: {col} ↔ {target_col} = {corr:.3f}"
                    )
                    leakage_report["has_leakage"] = True
                    leakage_report["recommendations"].append(
                        f"Supprimer ou modifier {col} (risque de leakage)"
                    )
                elif abs(corr) > 0.7:
                    leakage_report["warnings"].append(
                        f"Corrélation élevée: {col} ↔ {target_col} = {corr:.3f}"
                    )
    
    # 3. Vérifier la variance nulle (features inutiles)
    for col in feature_cols:
        if col in df.columns and df[col].nunique() == 1:
            leakage_report["warnings"].append(f"Feature constante: {col}")
            leakage_report["recommendations"].append(f"Supprimer {col} (aucune variance)")
    
    # 4. Vérifier les NaN excessifs
    for col in feature_cols:
        if col in df.columns:
            nan_ratio = df[col].isna().sum() / len(df)
            if nan_ratio > 0.5:
                leakage_report["warnings"].append(
                    f"Feature avec {nan_ratio:.1%} de NaN: {col}"
                )
    
    return leakage_report


# =============================================================================
# MAIN - Exemple d'utilisation
# =============================================================================

if __name__ == "__main__":
    # Exemple d'utilisation
    print("Module feature_engineer chargé avec succès")
    print("Classes disponibles:")
    print("  - FeatureEngineer: Rolling averages, stats domicile/extérieur")
    print("  - FootballELO: Système ELO avec margin of victory")
    print("  - TripleLayerFeatures: Divergences Bookmaker vs Polymarket")
    print("\nFonctions disponibles:")
    print("  - compute_xg_proxy(): Expected goals proxy")
    print("  - compute_fatigue_features(): Facteurs de fatigue")
    print("  - compute_h2h_features(): Head-to-Head")
    print("  - add_odds_features(): Probabilités bookmaker")
    print("  - check_data_leakage(): Vérification anti-leakage")
