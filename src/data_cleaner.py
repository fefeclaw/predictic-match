"""
Football Data Cleaner Module

This module provides the DataCleaner class for preprocessing and cleaning
football match data loaded from football-data.co.uk CSV files.

The class handles:
- Missing value management
- Result encoding (H=2, D=1, A=0)
- Feature engineering
- Data type conversion
- Outlier detection
"""

import os
from typing import List, Optional, Dict, Tuple, Union
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    A class for cleaning and preprocessing football match data.
    
    This class handles all data preprocessing steps required before
    feeding data into machine learning models for match prediction.
    
    Attributes:
        encoding_map (Dict): Mapping for full-time results (H=2, D=1, A=0)
        numeric_columns (List): List of numeric feature columns
        categorical_columns (List): List of categorical columns
        
    Example:
        >>> cleaner = DataCleaner()
        >>> df_clean = cleaner.clean_data(df_raw)
        >>> df_encoded = cleaner.encode_results(df_clean)
    """
    
    # Full-time result encoding
    RESULT_ENCODING: Dict[str, int] = {
        'H': 2,  # Home win
        'D': 1,  # Draw
        'A': 0   # Away win
    }
    
    # Reverse encoding for predictions
    RESULT_DECODING: Dict[int, str] = {
        2: 'H',
        1: 'D',
        0: 'A'
    }
    
    # Standard numeric columns in football-data.co.uk datasets
    STANDARD_NUMERIC_COLS: List[str] = [
        'FTHG',  # Full-time home goals
        'FTAG',  # Full-time away goals
        'FTHS',  # Full-time home shots
        'FTAS',  # Full-time away shots
        'HST',   # Home shots on target
        'AST',   # Away shots on target
        'HF',    # Home fouls
        'AF',    # Away fouls
        'HC',    # Home corners
        'AC',    # Away corners
        'HY',    # Home yellow cards
        'AY',    # Away yellow cards
        'HR',    # Home red cards
        'AR',    # Away red cards
    ]
    
    # Standard odds columns
    ODDS_COLUMNS: List[str] = [
        'B365H', 'B365D', 'B365A',  # Bet365 odds
        'BWH', 'BWD', 'BWA',        # Bet&Win odds
        'IWH', 'IWD', 'IWA',        # Interwetten odds
        'PSH', 'PSD', 'PSA',        # Pinnacle Sports odds
        'WHH', 'WHD', 'WHA',        # William Hill odds
        'VCH', 'VCD', 'VCA',        # VC Bet odds
        'GBH', 'GBD', 'GBA',        # Gamebookers odds
    ]
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the DataCleaner.
        
        Args:
            random_state: Random seed for reproducibility.
                         Defaults to RANDOM_STATE env variable or 42
        """
        self.random_state = random_state or int(os.getenv('RANDOM_STATE', 42))
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        logger.info(
            f"DataCleaner initialized with random_state={self.random_state}"
        )
    
    def clean_data(
        self,
        df: pd.DataFrame,
        drop_na_threshold: float = 0.5,
        fill_numeric: bool = True
    ) -> pd.DataFrame:
        """
        Clean the raw football data by handling missing values and types.
        
        Args:
            df: Raw DataFrame to clean
            drop_na_threshold: Drop columns with more than this fraction of NaN
            fill_numeric: Whether to fill numeric NaN with median
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning data: {len(df)} rows, {len(df.columns)} columns")
        
        df_clean = df.copy()
        
        # Store initial missing value stats
        initial_missing = df_clean.isnull().sum().sum()
        logger.info(f"Initial missing values: {initial_missing}")
        
        # Drop columns with too many missing values
        cols_to_drop = []
        for col in df_clean.columns:
            na_ratio = df_clean[col].isnull().mean()
            if na_ratio > drop_na_threshold:
                cols_to_drop.append(col)
                logger.debug(f"Dropping column {col} ({na_ratio:.2%} missing)")
        
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns with high missing rate")
        
        # Handle numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_numeric and df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                logger.debug(f"Filled {col} NaN with median={median_val}")
        
        # Handle categorical columns - fill with mode or 'Unknown'
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
                logger.debug(f"Filled {col} NaN with mode")
        
        # Convert numeric columns to proper types
        for col in numeric_cols:
            if col in ['FTHG', 'FTAG']:  # Goals should be integer
                df_clean[col] = df_clean[col].astype(int)
            elif col in self.STANDARD_NUMERIC_COLS:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        final_missing = df_clean.isnull().sum().sum()
        logger.info(
            f"Cleaning complete: {initial_missing} -> {final_missing} missing values"
        )
        
        return df_clean
    
    def encode_results(
        self,
        df: pd.DataFrame,
        column: str = 'FTR',
        new_column: str = 'FTR_encoded'
    ) -> pd.DataFrame:
        """
        Encode full-time results to numeric values (H=2, D=1, A=0).
        
        Args:
            df: DataFrame with result column
            column: Name of the result column (default: 'FTR')
            new_column: Name for the encoded column (default: 'FTR_encoded')
            
        Returns:
            DataFrame with encoded result column
            
        Raises:
            ValueError: If result column contains unexpected values
        """
        logger.info(f"Encoding results from column '{column}'")
        
        df_encoded = df.copy()
        
        if column not in df_encoded.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        # Check for unexpected values
        unique_values = set(df_encoded[column].unique())
        expected_values = {'H', 'D', 'A'}
        unexpected = unique_values - expected_values - {np.nan}
        
        if unexpected:
            logger.warning(f"Unexpected result values: {unexpected}")
            # Map unexpected values to NaN
            df_encoded[column] = df_encoded[column].where(
                df_encoded[column].isin(expected_values),
                np.nan
            )
        
        # Apply encoding
        df_encoded[new_column] = df_encoded[column].map(self.RESULT_ENCODING)
        
        # Log encoding distribution
        value_counts = df_encoded[new_column].value_counts(dropna=False)
        logger.info(f"Encoded result distribution:\n{value_counts}")
        
        return df_encoded
    
    def decode_results(
        self,
        encoded_values: Union[pd.Series, np.ndarray, List[int]]
    ) -> Union[pd.Series, List[str]]:
        """
        Decode numeric results back to H/D/A format.
        
        Args:
            encoded_values: Numeric encoded values (0, 1, 2)
            
        Returns:
            Decoded result strings ('H', 'D', 'A')
        """
        if isinstance(encoded_values, pd.Series):
            return encoded_values.map(self.RESULT_DECODING)
        elif isinstance(encoded_values, np.ndarray):
            return pd.Series(encoded_values).map(self.RESULT_DECODING).values
        else:
            return [self.RESULT_DECODING.get(v, 'Unknown') for v in encoded_values]
    
    def encode_teams(
        self,
        df: pd.DataFrame,
        home_col: str = 'HomeTeam',
        away_col: str = 'AwayTeam'
    ) -> pd.DataFrame:
        """
        Encode team names using LabelEncoder.
        
        Creates separate encoders for home and away teams to handle
        the categorical nature of team names.
        
        Args:
            df: DataFrame with team columns
            home_col: Name of home team column
            away_col: Name of away team column
            
        Returns:
            DataFrame with encoded team columns
        """
        logger.info(f"Encoding teams from '{home_col}' and '{away_col}'")
        
        df_encoded = df.copy()
        
        # Fit encoders on all teams (home + away)
        all_teams = pd.concat([df[home_col], df[away_col]]).unique()
        
        # Home team encoder
        self.label_encoders['home_team'] = LabelEncoder()
        self.label_encoders['home_team'].fit(all_teams)
        
        # Away team encoder (same classes)
        self.label_encoders['away_team'] = LabelEncoder()
        self.label_encoders['away_team'].fit(all_teams)
        
        # Apply encoding
        df_encoded['HomeTeam_enc'] = self.label_encoders['home_team'].transform(
            df[home_col]
        )
        df_encoded['AwayTeam_enc'] = self.label_encoders['away_team'].transform(
            df[away_col]
        )
        
        logger.info(
            f"Encoded {len(all_teams)} unique teams "
            f"(Home: {df_encoded['HomeTeam_enc'].nunique()}, "
            f"Away: {df_encoded['AwayTeam_enc'].nunique()})"
        )
        
        return df_encoded
    
    def create_features(
        self,
        df: pd.DataFrame,
        add_goal_difference: bool = True,
        add_total_goals: bool = True,
        add_shot_difference: bool = True,
        add_odds_features: bool = True
    ) -> pd.DataFrame:
        """
        Create derived features from raw data.
        
        Args:
            df: Cleaned DataFrame
            add_goal_difference: Create home-away goal difference
            add_total_goals: Create total goals feature
            add_shot_difference: Create shot difference feature
            add_odds_features: Create odds-based features
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Creating derived features")
        
        df_features = df.copy()
        features_created = []
        
        # Goal-based features
        if add_goal_difference and 'FTHG' in df_features.columns:
            df_features['GoalDiff'] = (
                df_features['FTHG'] - df_features['FTAG']
            )
            features_created.append('GoalDiff')
        
        if add_total_goals and 'FTHG' in df_features.columns:
            df_features['TotalGoals'] = (
                df_features['FTHG'] + df_features['FTAG']
            )
            features_created.append('TotalGoals')
        
        # Shot-based features
        if add_shot_difference:
            if 'FTHS' in df_features.columns and 'FTAS' in df_features.columns:
                df_features['ShotDiff'] = (
                    df_features['FTHS'] - df_features['FTAS']
                )
                features_created.append('ShotDiff')
            
            if 'HST' in df_features.columns and 'AST' in df_features.columns:
                df_features['ShotOnTargetDiff'] = (
                    df_features['HST'] - df_features['AST']
                )
                features_created.append('ShotOnTargetDiff')
        
        # Odds-based features
        if add_odds_features:
            # Use primary odds provider (Bet365 or first available)
            odds_cols = ['B365H', 'B365D', 'B365A']
            if all(col in df_features.columns for col in odds_cols):
                # Implied probabilities
                df_features['ImpliedProbHome'] = 1 / df_features['B365H']
                df_features['ImpliedProbDraw'] = 1 / df_features['B365D']
                df_features['ImpliedProbAway'] = 1 / df_features['B365A']
                
                # Normalize to sum to 1 (remove bookmaker margin)
                total_implied = (
                    df_features['ImpliedProbHome'] +
                    df_features['ImpliedProbDraw'] +
                    df_features['ImpliedProbAway']
                )
                df_features['NormProbHome'] = (
                    df_features['ImpliedProbHome'] / total_implied
                )
                df_features['NormProbDraw'] = (
                    df_features['ImpliedProbDraw'] / total_implied
                )
                df_features['NormProbAway'] = (
                    df_features['ImpliedProbAway'] / total_implied
                )
                
                # Odds difference
                df_features['OddsDiff'] = (
                    df_features['B365H'] - df_features['B365A']
                )
                
                features_created.extend([
                    'ImpliedProbHome', 'ImpliedProbDraw', 'ImpliedProbAway',
                    'NormProbHome', 'NormProbDraw', 'NormProbAway', 'OddsDiff'
                ])
        
        logger.info(f"Created {len(features_created)} new features: {features_created}")
        
        return df_features
    
    def select_features(
        self,
        df: pd.DataFrame,
        feature_list: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Select specific features for modeling.
        
        Args:
            df: DataFrame with all features
            feature_list: List of columns to keep. If None, uses defaults
            exclude_columns: List of columns to exclude
            
        Returns:
            DataFrame with selected features
        """
        logger.info("Selecting features for modeling")
        
        # Default features for prediction
        if feature_list is None:
            feature_list = [
                'HomeTeam_enc', 'AwayTeam_enc',
                'FTHG', 'FTAG',
                'FTHS', 'FTAS',
                'HST', 'AST',
                'HF', 'AF',
                'HC', 'AC',
                'HY', 'AY',
                'GoalDiff', 'TotalGoals',
                'ShotDiff', 'ShotOnTargetDiff',
            ]
            
            # Add odds features if available
            odds_features = [
                'NormProbHome', 'NormProbDraw', 'NormProbAway', 'OddsDiff'
            ]
            for feat in odds_features:
                if feat in df.columns:
                    feature_list.append(feat)
        
        # Filter to available columns
        available_features = [
            f for f in feature_list if f in df.columns
        ]
        
        # Exclude specified columns
        if exclude_columns:
            available_features = [
                f for f in available_features if f not in exclude_columns
            ]
        
        logger.info(
            f"Selected {len(available_features)} features "
            f"(excluded {len(feature_list) - len(available_features)})"
        )
        
        return df[available_features].copy()
    
    def prepare_modeling_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'FTR_encoded',
        drop_na_target: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final feature matrix and target vector for modeling.
        
        Args:
            df: Cleaned and processed DataFrame
            target_column: Name of target column
            drop_na_target: Whether to drop rows with NaN target
            
        Returns:
            Tuple of (feature_matrix, target_vector)
        """
        logger.info("Preparing data for modeling")
        
        df_prep = df.copy()
        
        # Drop rows with NaN target
        if drop_na_target and target_column in df_prep.columns:
            initial_rows = len(df_prep)
            df_prep = df_prep.dropna(subset=[target_column])
            dropped = initial_rows - len(df_prep)
            if dropped > 0:
                logger.info(f"Dropped {dropped} rows with NaN target")
        
        # Separate features and target
        feature_cols = [
            col for col in df_prep.columns
            if col != target_column and col not in [
                'Date', 'HomeTeam', 'AwayTeam', 'FTR',
                'season_str', 'league_name', 'Referee'
            ]
        ]
        
        X = df_prep[feature_cols]
        y = df_prep[target_column]
        
        # Convert target to integer
        y = y.astype(int)
        
        logger.info(
            f"Modeling data ready: X={X.shape}, y={y.shape}, "
            f"classes={y.nunique()}"
        )
        
        return X, y
    
    def get_encoding_info(self) -> Dict:
        """
        Get information about encodings used.
        
        Returns:
            Dictionary with encoding mappings
        """
        return {
            'result_encoding': self.RESULT_ENCODING,
            'result_decoding': self.RESULT_DECODING,
            'label_encoders': {
                name: list(encoder.classes_)
                for name, encoder in self.label_encoders.items()
            }
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'HomeTeam': ['Team A', 'Team B', 'Team C'],
        'AwayTeam': ['Team B', 'Team C', 'Team A'],
        'FTR': ['H', 'D', 'A'],
        'FTHG': [2, 1, 0],
        'FTAG': [1, 1, 2],
        'FTHS': [10, 8, 5],
        'FTAS': [5, 7, 12],
    })
    
    cleaner = DataCleaner()
    
    # Clean data
    df_clean = cleaner.clean_data(sample_data)
    
    # Encode results
    df_encoded = cleaner.encode_results(df_clean)
    
    # Encode teams
    df_encoded = cleaner.encode_teams(df_encoded)
    
    # Create features
    df_features = cleaner.create_features(df_encoded)
    
    print(df_features)
