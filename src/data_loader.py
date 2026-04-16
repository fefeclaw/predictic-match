"""
Football Data Loader Module

This module provides the FootballDataLoader class for loading and managing
football match data from football-data.co.uk CSV files.

The class supports:
- Multiple leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1, etc.)
- Multiple seasons
- Automatic data validation and type inference
- Efficient memory management for large datasets
"""

import os
import glob
from pathlib import Path
from typing import List, Optional, Union, Dict
import logging

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class FootballDataLoader:
    """
    A class for loading football match data from football-data.co.uk CSV files.
    
    This class handles the loading, concatenation, and basic validation of 
    historical football match data across multiple leagues and seasons.
    
    Attributes:
        raw_data_path (str): Path to the raw data directory
        leagues (List[str]): List of league codes to load
        seasons (List[int]): List of seasons to load
        
    Example:
        >>> loader = FootballDataLoader(
        ...     raw_data_path='data/raw',
        ...     leagues=['E0', 'SP1'],
        ...     seasons=[2020, 2021, 2022]
        ... )
        >>> df = loader.load_all_data()
    """
    
    # League codes mapping (football-data.co.uk convention)
    LEAGUE_MAPPING: Dict[str, str] = {
        'E0': 'Premier League',
        'E1': 'Championship',
        'E2': 'League One',
        'E3': 'League Two',
        'E4': 'Conference',
        'SP1': 'La Liga',
        'SP2': 'Segunda División',
        'I1': 'Serie A',
        'I2': 'Serie B',
        'D1': 'Bundesliga',
        'D2': '2. Bundesliga',
        'F1': 'Ligue 1',
        'F2': 'Ligue 2',
        'N1': 'Eredivisie',
        'P1': 'Primeira Liga',
        'B1': 'Belgian Pro League',
        'T1': 'Süper Lig',
        'G1': 'Super League Greece',
        'R1': 'Russian Premier League',
        'A1': 'Austrian Bundesliga',
        'S1': 'Swiss Super League',
        'C1': 'UEFA Champions League',
        'EL': 'UEFA Europa League',
    }
    
    def __init__(
        self,
        raw_data_path: Optional[str] = None,
        leagues: Optional[List[str]] = None,
        seasons: Optional[List[int]] = None
    ):
        """
        Initialize the FootballDataLoader.
        
        Args:
            raw_data_path: Path to the raw data directory. 
                          Defaults to DATA_RAW_PATH env variable or 'data/raw'
            leagues: List of league codes to load (e.g., ['E0', 'SP1']).
                    Defaults to ['E0'] (Premier League)
            seasons: List of seasons to load (e.g., [2020, 2021, 2022]).
                    Defaults to seasons from SEASONS_START to SEASONS_END env vars
        """
        # Set raw data path
        self.raw_data_path = raw_data_path or os.getenv('DATA_RAW_PATH', 'data/raw')
        
        # Set leagues
        if leagues is None:
            default_league = os.getenv('DEFAULT_LEAGUE', 'E0')
            self.leagues = [default_league]
        else:
            self.leagues = leagues
            # Validate league codes
            for league in self.leagues:
                if league not in self.LEAGUE_MAPPING:
                    logger.warning(f"Unknown league code: {league}")
        
        # Set seasons
        if seasons is None:
            start = int(os.getenv('SEASONS_START', 2018))
            end = int(os.getenv('SEASONS_END', 2024))
            self.seasons = list(range(start, end + 1))
        else:
            self.seasons = seasons
        
        logger.info(
            f"FootballDataLoader initialized - "
            f"Path: {self.raw_data_path}, "
            f"Leagues: {self.leagues}, "
            f"Seasons: {self.seasons}"
        )
    
    def _get_season_suffix(self, season: int) -> str:
        """
        Convert a season year to football-data.co.uk file suffix format.
        
        Args:
            season: The starting year of the season (e.g., 2020 for 2020-2021)
            
        Returns:
            String suffix for the season (e.g., '20' for 2020)
            
        Example:
            >>> loader._get_season_suffix(2020)
            '20'
            >>> loader._get_season_suffix(2019)
            '19'
        """
        # football-data.co.uk uses last 2 digits of the starting year
        return str(season)[-2:]
    
    def _build_file_path(self, league: str, season: int) -> str:
        """
        Build the expected file path for a league-season combination.
        
        Args:
            league: League code (e.g., 'E0')
            season: Season year (e.g., 2020)
            
        Returns:
            Full file path for the CSV file
        """
        suffix = self._get_season_suffix(season)
        filename = f"{league.lower()}{suffix}.csv"
        return os.path.join(self.raw_data_path, filename)
    
    def load_single_file(
        self,
        file_path: str,
        league: Optional[str] = None,
        season: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load a single CSV file and add metadata columns.
        
        Args:
            file_path: Path to the CSV file
            league: Optional league code to add as metadata
            season: Optional season year to add as metadata
            
        Returns:
            DataFrame with match data and metadata columns
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
        """
        logger.debug(f"Loading file: {file_path}")
        
        # Load CSV with optimized settings
        df = pd.read_csv(
            file_path,
            encoding='utf-8',
            low_memory=False,
            skipinitialspace=True
        )
        
        # Add metadata columns if provided
        if league is not None:
            df['league_code'] = league
            df['league_name'] = self.LEAGUE_MAPPING.get(league, 'Unknown')
        
        if season is not None:
            df['season'] = season
            # Create season string (e.g., '2020-2021')
            df['season_str'] = f"{season}-{season + 1}"
        
        logger.debug(f"Loaded {len(df)} rows from {file_path}")
        
        return df
    
    def load_league_season(
        self,
        league: str,
        season: int
    ) -> Optional[pd.DataFrame]:
        """
        Load data for a specific league and season.
        
        Args:
            league: League code (e.g., 'E0')
            season: Season year (e.g., 2020)
            
        Returns:
            DataFrame with match data, or None if file not found
        """
        file_path = self._build_file_path(league, season)
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            df = self.load_single_file(file_path, league=league, season=season)
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return None
    
    def load_all_data(
        self,
        ignore_missing: bool = True
    ) -> pd.DataFrame:
        """
        Load all data for configured leagues and seasons.
        
        Args:
            ignore_missing: If True, skip missing files. If False, raise error.
            
        Returns:
            Concatenated DataFrame with all match data
            
        Raises:
            FileNotFoundError: If ignore_missing=False and a file is missing
            ValueError: If no data files are found
        """
        dataframes = []
        missing_files = []
        
        for league in self.leagues:
            for season in self.seasons:
                file_path = self._build_file_path(league, season)
                
                if os.path.exists(file_path):
                    try:
                        df = self.load_single_file(
                            file_path,
                            league=league,
                            season=season
                        )
                        dataframes.append(df)
                        logger.info(
                            f"Loaded: {league} {season}-{season+1} "
                            f"({len(df)} matches)"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error loading {file_path}: {str(e)}"
                        )
                        if not ignore_missing:
                            raise
                else:
                    missing_files.append(file_path)
                    if not ignore_missing:
                        raise FileNotFoundError(f"Missing file: {file_path}")
        
        if not dataframes:
            error_msg = "No data files found for the specified leagues and seasons"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        logger.info(
            f"Combined dataset: {len(combined_df)} matches, "
            f"{len(combined_df.columns)} columns"
        )
        
        if missing_files:
            logger.warning(
                f"Skipped {len(missing_files)} missing files. "
                f"Use ignore_missing=False to raise errors."
            )
        
        return combined_df
    
    def load_from_glob(
        self,
        pattern: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load all CSV files matching a glob pattern.
        
        This is useful for loading all available files without specifying
        leagues and seasons explicitly.
        
        Args:
            pattern: Glob pattern for file matching.
                    Defaults to '*.csv' in raw_data_path
            
        Returns:
            Concatenated DataFrame with all matching files
        """
        if pattern is None:
            pattern = os.path.join(self.raw_data_path, '*.csv')
        
        files = glob.glob(pattern)
        
        if not files:
            raise ValueError(f"No files found matching pattern: {pattern}")
        
        dataframes = []
        
        for file_path in sorted(files):
            try:
                # Extract league and season from filename
                filename = os.path.basename(file_path)
                # Pattern: e020.csv -> league='E0', season=2020
                if len(filename) >= 6:
                    league_code = filename[:2].upper()
                    season_suffix = filename[2:4]
                    # Handle seasons from 2000s and 1990s
                    if season_suffix.isdigit():
                        season_year = int(season_suffix)
                        if season_year > 50:  # 1990s
                            season_year += 1900
                        else:  # 2000s
                            season_year += 2000
                    else:
                        season_year = None
                        league_code = None
                else:
                    season_year = None
                    league_code = None
                
                df = self.load_single_file(
                    file_path,
                    league=league_code,
                    season=season_year
                )
                dataframes.append(df)
                logger.info(f"Loaded: {filename} ({len(df)} matches)")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                continue
        
        if not dataframes:
            raise ValueError("No valid data files could be loaded")
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} total matches from {len(files)} files")
        
        return combined_df
    
    def get_league_info(self) -> Dict[str, str]:
        """
        Get the mapping of league codes to league names.
        
        Returns:
            Dictionary mapping league codes to full names
        """
        return self.LEAGUE_MAPPING.copy()
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform basic validation on loaded data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicate_rows': 0,
            'issues': []
        }
        
        # Check for missing values
        missing = df.isnull().sum()
        validation['missing_values'] = missing[missing > 0].to_dict()
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        validation['duplicate_rows'] = int(duplicates)
        
        if duplicates > 0:
            validation['issues'].append(f"Found {duplicates} duplicate rows")
        
        # Check for required columns
        required_columns = ['HomeTeam', 'AwayTeam', 'FTR']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            validation['is_valid'] = False
            validation['issues'].append(f"Missing required columns: {missing_cols}")
        
        # Check for empty dataframe
        if len(df) == 0:
            validation['is_valid'] = False
            validation['issues'].append("DataFrame is empty")
        
        logger.info(
            f"Validation complete: {validation['is_valid']} - "
            f"{len(validation['issues'])} issues found"
        )
        
        return validation


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    loader = FootballDataLoader(
        raw_data_path='data/raw',
        leagues=['E0'],
        seasons=[2020, 2021, 2022]
    )
    
    # Load all data
    df = loader.load_all_data(ignore_missing=True)
    
    # Validate
    validation = loader.validate_data(df)
    print(f"Validation: {validation}")
