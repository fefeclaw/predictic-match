"""
Predictic Match - Football Match Prediction Package

A comprehensive package for loading, cleaning, and processing football data
from football-data.co.uk for match prediction modeling.

Modules:
    data_loader: FootballDataLoader class for CSV data loading
    data_cleaner: DataCleaner class for data preprocessing
"""

__version__ = "0.1.0"
__author__ = "Predictic Match Team"

from .data_loader import FootballDataLoader
from .data_cleaner import DataCleaner

__all__ = ["FootballDataLoader", "DataCleaner"]
