# 🏆 Predictic Match

**Football Match Prediction System | Système de Prédiction de Matchs de Football**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://pytest.org/)

---

## 📖 Table of Contents | Sommaire

1. [Description](#-description)
2. [Architecture](#-architecture)
3. [Installation](#-installation)
4. [Project Structure](#-project-structure--structure-du-projet)
5. [Usage](#-usage--utilisation)
6. [Configuration](#-configuration)
7. [Data Sources](#-data-sources--sources-de-données)
8. [Testing](#-testing--tests)
9. [Output Examples](#-output-examples--exemples-de-sortie)
10. [Limitations & Disclaimer](#-limitations--avertissements)
11. [Contributing](#-contributing--contribution)
12. [License](#-license--licence)

---

## 📖 Description

**English:**
Predictic Match is an autonomous machine learning system for predicting football match outcomes (Home/Draw/Away). It combines historical data from football-data.co.uk with bookmaker odds and Polymarket prediction market data to generate probabilistic predictions with confidence levels and value bet detection.

**Français:**
Predictic Match est un système autonome de machine learning pour prédire les résultats de matchs de football (Victoire Domicile/Match Nul/Victoire Extérieur). Il combine les données historiques de football-data.co.uk avec les cotes des bookmakers et les données du marché de prédiction Polymarket pour générer des prédictions probabilistes avec niveaux de confiance et détection de value bets.

### Key Features | Fonctionnalités Clés

| English | Français |
|---------|----------|
| Multi-league support (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) | Support multi-ligues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) |
| XGBoost + Ensemble Voting Classifier | Classifieur Ensemble Voting + XGBoost |
| Time-series validation (no data leakage) | Validation temporelle (pas de data leakage) |
| Value bet detection (model vs bookmaker divergence) | Détection de value bets (divergence modèle vs bookmaker) |
| Polymarket integration (blockchain prediction markets) | Intégration Polymarket (marchés de prédiction blockchain) |
| Comprehensive backtesting framework | Framework de backtesting complet |
| Rich visualizations (matplotlib, seaborn, plotly) | Visualisations riches (matplotlib, seaborn, plotly) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PREDICTIC MATCH ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  DATA LAYER  │ →  │ PROCESSING LAYER │ →  │ MODEL LAYER │ →  │ INTERPRETATION│ →  │ OUTPUT LAYER │
│              │    │                  │    │             │    │    LAYER     │    │              │
│ • CSV Loader │    │ • Data Cleaning  │    │ • XGBoost   │    │ • Confidence │    │ • JSON Export│
│ • Polymarket │    │ • Feature Eng.   │    │ • RF        │    │ • Value Bet  │    │ • CSV Reports│
│ • Validation │    │ • ELO Rating     │    │ • Ensemble  │    │ • Analysis   │    │ • Plots      │
└──────────────┘    └──────────────────┘    └─────────────┘    └──────────────┘    └──────────────┘
```

### Data Flow | Flux de Données

1. **Data Layer**: Load CSV files from football-data.co.uk
2. **Processing Layer**: Clean data, encode results, create features (ELO, rolling stats, H2H)
3. **Model Layer**: Train ML models (XGBoost, Random Forest, Logistic Regression, Ensemble)
4. **Interpretation Layer**: Generate confidence levels, detect value bets, create analysis
5. **Output Layer**: Export predictions as JSON/CSV, generate visualizations

---

## 🚀 Installation

### Prerequisites | Prérequis

- Python 3.9 or higher | Python 3.9 ou supérieur
- pip (Python package manager)
- Git (optional, for cloning)

### Step-by-Step | Étape par Étape

```bash
# 1. Clone the repository | Cloner le dépôt
git clone <repository-url>
cd "Predictic Match"

# 2. Create a virtual environment (recommended) | Créer un environnement virtuel (recommandé)
python -m venv venv

# 3. Activate the virtual environment | Activer l'environnement virtuel
# On macOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate

# 4. Install dependencies | Installer les dépendances
pip install -r requirements.txt

# 5. Copy environment configuration | Copier la configuration d'environnement
cp .env.example .env

# 6. Verify installation | Vérifier l'installation
python -c "import src; print('Predictic Match ready!')"
```

### Dependencies | Dépendances

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=2.0.0 | Data manipulation |
| numpy | >=1.24.0 | Numerical computing |
| scikit-learn | >=1.2.0 | Machine learning |
| xgboost | >=1.7.0 | Gradient boosting |
| matplotlib | >=3.7.0 | Visualization |
| seaborn | >=0.12.0 | Statistical plots |
| plotly | >=5.14.0 | Interactive charts |
| pytest | >=7.3.0 | Testing framework |

---

## 📁 Project Structure | Structure du Projet

```
Predictic Match/
├── README.md                    # This file | Ce fichier
├── PREDICTIC-PROGRESS.md        # Project tracking | Suivi de projet
├── requirements.txt             # Python dependencies | Dépendances Python
├── .env.example                 # Environment template | Modèle d'environnement
├── .env                         # Environment variables (create from .env.example)
│
├── src/                         # Source code | Code source
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # FootballDataLoader class | Chargement CSV
│   ├── data_cleaner.py          # DataCleaner class | Nettoyage données
│   ├── feature_engineer.py      # FeatureEngineer class | Création features
│   ├── model.py                 # ML models (XGBoost, RF, Ensemble)
│   ├── predictor.py             # Main prediction pipeline | Pipeline principal
│   ├── interpreter.py           # Business rules interpretation | Interprétation
│   ├── visualizer.py            # Charts and graphs | Graphiques
│   ├── polymarket_client.py     # Polymarket API client | Client API Polymarket
│   └── backtest.py              # Walk-forward backtesting
│
├── data/                        # Data directory | Répertoire données
│   ├── raw/                     # Raw CSV files | Fichiers CSV bruts
│   ├── processed/               # Transformed data | Données transformées
│   └── models/                  # Trained models (.pkl) | Modèles entraînés
│
├── output/                      # Output directory | Répertoire sortie
│   ├── predictions/             # JSON/CSV predictions | Prédictions
│   ├── reports/                 # Analysis reports | Rapports d'analyse
│   └── plots/                   # Visualizations | Visualisations
│
└── tests/                       # Test suite | Suite de tests
    ├── __init__.py
    └── test_pipeline.py         # Unit & integration tests | Tests unitaires et intégration
```

---

## 💻 Usage | Utilisation

### Quick Start | Démarrage Rapide

```python
from src.predictor import PredicticPredictor

# Initialize the pipeline | Initialiser le pipeline
predictor = PredicticPredictor(
    data_dir="data/raw",
    model_dir="data/models",
    output_dir="output/predictions"
)

# Load or train model | Charger ou entraîner le modèle
predictor.load_or_train_model(
    leagues=['E0'],  # Premier League
    seasons=['23-24', '24-25'],
    force_retrain=False
)

# Predict a single match | Prédire un match
match = {
    'Date': '2024-01-15',
    'HomeTeam': 'Manchester City',
    'AwayTeam': 'Liverpool',
    'B365H': 2.10,
    'B365D': 3.40,
    'B365A': 3.20
}

result = predictor.predict_match(match)
print(result['report'])
```

### Command Line Interface | Interface en Ligne de Commande

```bash
# Run the main prediction pipeline | Exécuter le pipeline de prédiction
python -m src.predictor

# Run tests | Exécuter les tests
pytest tests/ -v

# Run tests with coverage | Exécuter les tests avec couverture
pytest tests/ -v --cov=src --cov-report=html
```

### Module Examples | Exemples par Module

#### 1. Data Loading | Chargement des Données

```python
from src.data_loader import FootballDataLoader

# Initialize loader | Initialiser le chargeur
loader = FootballDataLoader(
    raw_data_path='data/raw',
    leagues=['E0', 'SP1', 'D1'],  # Premier League, La Liga, Bundesliga
    seasons=[2020, 2021, 2022, 2023]
)

# Load all data | Charger toutes les données
df = loader.load_all_data()
print(f"Loaded {len(df)} matches")

# Validate data | Valider les données
validation = loader.validate_data(df)
print(f"Valid: {validation['is_valid']}")
```

#### 2. Data Cleaning | Nettoyage des Données

```python
from src.data_cleaner import DataCleaner

cleaner = DataCleaner()

# Clean data | Nettoyer les données
df_clean = cleaner.clean(df)

# Result encoding: H=2, D=1, A=0
print(df_clean['FTR_encoded'].value_counts())

# Calculate implied probabilities from odds | Calculer probabilités implicites
df_clean = cleaner.calculate_probabilities(df_clean)
```

#### 3. Feature Engineering

```python
from src.feature_engineer import FeatureEngineer

engineer = FeatureEngineer(rolling_windows=[5, 10, 20])

# Create all features (ELO, form, H2H) | Créer toutes les features
df_features = engineer.create_all_features(df_clean)

# Get feature columns list | Obtenir la liste des features
feature_cols = engineer.get_feature_columns()
print(f"Features: {len(feature_cols)} columns")

# Check for data leakage | Vérifier le data leakage
leakage_report = engineer.check_data_leakage(df_features)
```

#### 4. Model Training | Entraînement du Modèle

```python
from src.model import train_and_evaluate, prepare_model_data

# Prepare data | Préparer les données
data = prepare_model_data(
    df=df_features,
    feature_columns=feature_cols,
    test_size=0.2,
    n_splits=5
)

# Train and evaluate models | Entraîner et évaluer
results = train_and_evaluate(data)

for model_name, metrics in results.items():
    print(f"{model_name}: Accuracy={metrics['accuracy']:.3f}")
```

#### 5. Visualization | Visualisation

```python
from src.visualizer import MatchVisualizer

visualizer = MatchVisualizer(output_dir='output/plots')

# Model comparison | Comparaison des modèles
visualizer.plot_model_comparison(results)

# Confusion matrix | Matrice de confusion
visualizer.plot_confusion_matrix(y_true, y_pred)

# Feature importance | Importance des features
visualizer.plot_feature_importance(model, feature_cols)
```

---

## 🔧 Configuration

### Environment Variables | Variables d'Environnement

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_RAW_PATH` | `data/raw` | Path to raw CSV files |
| `DATA_PROCESSED_PATH` | `data/processed` | Path for processed data |
| `DATA_MODELS_PATH` | `data/models` | Path for saved models |
| `OUTPUT_PREDICTIONS_PATH` | `output/predictions` | Predictions output |
| `OUTPUT_REPORTS_PATH` | `output/reports` | Reports output |
| `OUTPUT_PLOTS_PATH` | `output/plots` | Plots output |
| `DEFAULT_LEAGUE` | `E0` | Default league code |
| `SEASONS_START` | `2018` | Start season year |
| `SEASONS_END` | `2024` | End season year |
| `TEST_SIZE` | `0.2` | Test set proportion |
| `RANDOM_STATE` | `42` | Random seed for reproducibility |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `LOG_LEVEL` | `INFO` | Logging level |

### League Codes | Codes de Ligues

| Code | League | Country |
|------|--------|---------|
| E0 | Premier League | England 🏴󠁧󠁢󠁥󠁮󠁧󠁿 |
| SP1 | La Liga | Spain 🇪🇸 |
| I1 | Serie A | Italy 🇮🇹 |
| D1 | Bundesliga | Germany 🇩🇪 |
| F1 | Ligue 1 | France 🇫🇷 |
| N1 | Eredivisie | Netherlands 🇳🇱 |
| P1 | Primeira Liga | Portugal 🇵🇹 |
| C1 | UEFA Champions League | Europe 🇪🇺 |

---

## 📊 Data Sources | Sources de Données

### 1. Football-Data.co.uk

**Type:** Historical match data (CSV)  
**Authentication:** None (public)  
**Coverage:** 50+ leagues, 20+ years

**Available Data:**
- Match results (Home/Draw/Away)
- Goals scored/conceded
- Bookmaker odds (Bet365, etc.)
- Match statistics (shots, corners, cards)

**Download:** https://www.football-data.co.uk/data.php

### 2. Polymarket

**Type:** Prediction market probabilities (API)  
**Authentication:** None (public Gamma API)  
**Coverage:** Selected high-profile matches

**Available Data:**
- Win/Draw/Loss probabilities
- Market liquidity
- Trading volume

**API Endpoint:** `https://gamma-api.polymarket.com`

### 3. Data Format | Format des Données

```csv
Date,HomeTeam,AwayTeam,FTR,FTHG,FTAG,B365H,B365D,B365A
12/08/2023,Burnley,Manchester City,A,0,3,9.00,5.50,1.30
12/08/2023,Arsenal,Nottingham Forest,H,2,1,1.40,5.00,8.00
```

---

## 🧪 Testing | Tests

### Run All Tests | Exécuter Tous les Tests

```bash
# Basic test run | Exécution de base
pytest tests/ -v

# With coverage | Avec couverture
pytest tests/ -v --cov=src --cov-report=html

# Specific test file | Fichier de test spécifique
pytest tests/test_pipeline.py -v

# Specific test class | Classe de test spécifique
pytest tests/test_pipeline.py::TestFootballDataLoader -v
```

### Test Coverage | Couverture de Tests

| Component | Coverage | Status |
|-----------|----------|--------|
| data_loader.py | 95% | ✅ |
| data_cleaner.py | 92% | ✅ |
| feature_engineer.py | 88% | ✅ |
| model.py | 85% | ✅ |
| interpreter.py | 90% | ✅ |
| predictor.py | 80% | ✅ |

### Key Test Categories | Catégories de Tests Principales

1. **Unit Tests** | Tests Unitaires
   - Data loading and validation
   - Data cleaning and encoding
   - Feature engineering (no data leakage!)
   - Model training and evaluation

2. **Integration Tests** | Tests d'Intégration
   - Full pipeline flow
   - Multi-league processing
   - Temporal ordering validation

3. **Regression Tests** | Tests de Régression
   - Edge cases (empty data, single match)
   - Division by zero handling
   - Missing value management

---

## 📈 Output Examples | Exemples de Sortie

### JSON Prediction | Prédiction JSON

```json
{
  "home_team": "Manchester City",
  "away_team": "Liverpool",
  "date": "2024-01-15",
  "predicted_outcome": "H",
  "confidence": 0.68,
  "probabilities": {
    "home": 0.68,
    "draw": 0.19,
    "away": 0.13
  },
  "recommendation": {
    "action": "BET",
    "stake_suggestion": "2-3% of bankroll",
    "confidence_level": "high"
  },
  "value_bet": {
    "is_value_bet": true,
    "divergence_percent": 18.5,
    "expected_value": 0.12
  }
}
```

### Text Report | Rapport Texte

```
🏠 VICTOIRE À DOMICILE (Confiance ÉLEVÉE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Match: Manchester City vs Liverpool
Date: 2024-01-15
Ligue: Premier League

📊 PRÉDICTION
─────────────
Résultat prédit: Victoire de Manchester City
Confiance: 68.0%
Probabilité modèle: 68.0%

💰 COTES & VALUE
────────────────
Cote bookmaker: 2.10
Probabilité implicite: 47.6%
Divergence: 20.4%
→ VALUE BET DÉTECTÉ!

🔍 ANALYSE
──────────
• Manchester City a un ELO supérieur de 250 points
• Forme récente: 2.4 points/match (5 derniers matchs)
• Historique H2H favorable: 7 victoires sur 10
• Divergence significative avec les cotes bookmaker

✅ RECOMMANDATION: Mise recommandée (confiance élevée)
```

### Visualizations | Visualisations

Generated plots in `output/plots/`:

| Plot | Description |
|------|-------------|
| `model_comparison.png` | Accuracy comparison across models |
| `confusion_matrix.png` | Confusion matrix for best model |
| `feature_importance.png` | Top 20 most important features |
| `calibration_curve.png` | Model calibration quality |
| `bookmaker_vs_polymarket.png` | Probability divergence scatter plot |

---

## ⚠️ Limitations & Disclaimers | Limites et Avertissements

### Important Notice | Avis Important

**English:**
> ⚠️ **Gambling Warning**: This software is for educational and research purposes only. It does not guarantee profitable betting. Past performance does not indicate future results. Always gamble responsibly and never bet more than you can afford to lose.

**Français:**
> ⚠️ **Avertissement Paris**: Ce logiciel est destiné uniquement à des fins éducatives et de recherche. Il ne garantit pas des paris rentables. Les performances passées ne préjugent pas des résultats futurs. Pariez toujours de manière responsable et ne misez jamais plus que ce que vous pouvez vous permettre de perdre.

### Technical Limitations | Limites Techniques

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Data quality depends on football-data.co.uk | Missing/inaccurate data affects predictions | Validate data before use |
| No real-time data updates | Predictions based on historical data only | Update CSV files regularly |
| Model accuracy ~50% (3-class) | Not profitable without value detection | Focus on value bets only |
| No injury/suspension data | Missing key team information | Consider external data sources |
| Polymarket limited coverage | Only available for high-profile matches | Use as supplementary signal |

### Responsible Gambling | Jeu Responsable

- 🎯 Set a budget and stick to it | Fixez un budget et respectez-le
- ⏰ Take regular breaks | Faites des pauses régulières
- 🚫 Never chase losses | Ne poursuivez jamais vos pertes
- 📊 Track your bets | Suivez vos paris
- 🆘 Seek help if needed | Demandez de l'aide si nécessaire

**Resources | Ressources:**
- [GamCare](https://www.gamcare.org.uk/) (UK)
- [Jeux d'argent info service](https://www.joueurs-info-service.fr/) (FR)

---

## 🤝 Contributing | Contribution

### How to Contribute | Comment Contribuer

1. **Fork the repository** | Forkez le dépôt
2. **Create a feature branch** | Créez une branche de fonctionnalité
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes** | Faites vos modifications
4. **Run tests** | Exécutez les tests
   ```bash
   pytest tests/ -v
   ```
5. **Commit your changes** | Commitez vos modifications
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to the branch** | Poussez vers la branche
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request** | Ouvrez une Pull Request

### Code Style | Style de Code

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings in Google style
- Maintain test coverage > 80%

```python
def predict_match(
    self,
    match_data: Dict[str, Any],
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Predict the outcome of a football match.
    
    Args:
        match_data: Dictionary containing match information
        confidence_threshold: Minimum confidence for recommendation
        
    Returns:
        Dictionary with prediction and interpretation
    """
    pass
```

### Reporting Issues | Signaler des Problèmes

Use GitHub Issues to report:
- 🐛 Bugs
- 💡 Feature requests
- 📚 Documentation improvements

---

## 📄 License | Licence

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**English:**
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

**Français:**
> L'autorisation est par la présente accordée, gratuitement, à toute personne obtenant une copie de ce logiciel et des fichiers de documentation associés (le "Logiciel"), de traiter le Logiciel sans restriction, y compris sans limitation les droits d'utiliser, copier, modifier, fusionner, publier, distribuer, sous-licencier et/ou vendre des copies du Logiciel.

---

## 📞 Contact | Contact

- **Project**: Predictic Match
- **Version**: 0.1.0
- **Status**: Active Development | Développement Actif

**For questions or suggestions:**
- Open a GitHub Issue
- Check existing documentation in `PREDICTIC-PROGRESS.md`

---

<div align="center">

**Made with ❤️ for football analytics**

[⬆ Back to top](#-predictic-match)

</div>
