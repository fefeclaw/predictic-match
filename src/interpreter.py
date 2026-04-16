"""
Module d'interprétation des prédictions - Règles métier SANS Claude API
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


class PredictionInterpreter:
    """
    Interprète les prédictions du modèle en utilisant des règles métier.
    Aucun appel à des APIs externes (Claude, etc.) - 100% algorithmique.
    """
    
    # Seuils de confiance
    CONFIDENCE_THRESHOLDS = {
        'low': 0.40,      # < 40% : confiance faible
        'medium': 0.60,   # 40-60% : confiance moyenne
        'high': 1.0       # > 60% : confiance élevée
    }
    
    # Seuil de value bet (divergence bookmaker vs modèle)
    VALUE_BET_THRESHOLD = 0.15  # 15% de divergence
    
    # Templates de rapports
    REPORT_TEMPLATES = {
        'high_confidence_home': """
🏠 VICTOIRE À DOMICILE (Confiance ÉLEVÉE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Match: {home_team} vs {away_team}
Date: {match_date}
Ligue: {league}

📊 PRÉDICTION
─────────────
Résultat prédit: Victoire de {home_team}
Confiance: {confidence:.1f}%
Probabilité modèle: {model_prob:.1f}%

💰 COTES & VALUE
────────────────
Cote bookmaker: {bookmaker_odd:.2f}
Probabilité implicite: {implied_prob:.1f}%
Divergence: {divergence:.1f}%
→ {value_bet_status}

🔍 ANALYSE
──────────
{analysis_points}

✅ RECOMMANDATION: Mise recommandée (confiance élevée)
        """,
        
        'medium_confidence': """
⚽ PRÉDICTION (Confiance MOYENNE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Match: {home_team} vs {away_team}
Date: {match_date}
Ligue: {league}

📊 PRÉDICTION
─────────────
Résultat prédit: {predicted_outcome}
Confiance: {confidence:.1f}%
Probabilité modèle: {model_prob:.1f}%

💰 COTES & VALUE
────────────────
Cote bookmaker: {bookmaker_odd:.2f}
Probabilité implicite: {implied_prob:.1f}%
Divergence: {divergence:.1f}%
→ {value_bet_status}

🔍 ANALYSE
──────────
{analysis_points}

⚠️ RECOMMANDATION: Mise optionnelle (confiance moyenne)
        """,
        
        'low_confidence': """
📋 PRÉDICTION (Confiance FAIBLE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Match: {home_team} vs {away_team}
Date: {match_date}
Ligue: {league}

📊 PRÉDICTION
─────────────
Résultat prédit: {predicted_outcome}
Confiance: {confidence:.1f}%
Probabilité modèle: {model_prob:.1f}%

⚠️ ANALYSE
──────────
{analysis_points}

❌ RECOMMANDATION: Aucune mise (confiance insuffisante)
        """
    }
    
    def __init__(self):
        """Initialise l'interpréteur."""
        self.interpretation_history: List[Dict] = []
        
    def get_confidence_level(self, probability: float) -> Tuple[str, str]:
        """
        Détermine le niveau de confiance depuis une probabilité.
        
        Args:
            probability: Probabilité de la prédiction (0-1)
            
        Returns:
            Tuple (niveau, description)
        """
        if probability < self.CONFIDENCE_THRESHOLDS['low']:
            return ('low', 'Faible')
        elif probability < self.CONFIDENCE_THRESHOLDS['medium']:
            return ('medium', 'Moyenne')
        else:
            return ('high', 'Élevée')
    
    def detect_value_bet(
        self,
        model_probability: float,
        bookmaker_odd: float
    ) -> Dict:
        """
        Détecte les value bets en comparant modèle vs bookmaker.
        
        Un value bet existe quand:
        - La probabilité du modèle > probabilité implicite du bookmaker + seuil
        
        Args:
            model_probability: Probabilité estimée par le modèle
            bookmaker_odd: Cote du bookmaker
            
        Returns:
            Dictionnaire avec analyse du value bet
        """
        # Probabilité implicite depuis la cote
        implied_probability = 1.0 / bookmaker_odd
        
        # Divergence absolue
        divergence = model_probability - implied_probability
        
        # Divergence relative (%)
        divergence_percent = (divergence / implied_probability) * 100 if implied_probability > 0 else 0
        
        # Est-ce un value bet ?
        is_value_bet = divergence > self.VALUE_BET_THRESHOLD
        
        # Valeur attendue (Expected Value)
        expected_value = (model_probability * (bookmaker_odd - 1)) - (1 - model_probability)
        
        return {
            'is_value_bet': is_value_bet,
            'divergence_absolute': divergence,
            'divergence_percent': divergence_percent,
            'implied_probability': implied_probability,
            'expected_value': expected_value,
            'recommendation': 'VALUE BET DETECTÉ' if is_value_bet else 'Pas de value bet'
        }
    
    def generate_analysis_points(
        self,
        features: Dict,
        prediction: Dict
    ) -> List[str]:
        """
        Génère des points d'analyse basés sur les features.
        
        Args:
            features: Dictionnaire des features du match
            prediction: Dictionnaire de prédiction
            
        Returns:
            Liste de points d'analyse
        """
        points = []
        
        # Analyse ELO
        if 'elo_diff' in features:
            elo_diff = features['elo_diff']
            if elo_diff > 200:
                points.append(f"• Écart ELO important (+{elo_diff:.0f}) : avantage significatif")
            elif elo_diff > 100:
                points.append(f"• Écart ELO modéré (+{elo_diff:.0f}) : léger avantage")
            elif elo_diff < -200:
                points.append(f"• Écart ELO défavorable ({elo_diff:.0f}) : désavantage")
        
        # Analyse forme récente
        if 'home_form_5' in features:
            form = features['home_form_5']
            if form > 2.0:
                points.append(f"• Excellente forme récente ({form:.1f} pts/match)")
            elif form < 1.0:
                points.append(f"• Mauvaise forme récente ({form:.1f} pts/match)")
        
        # Analyse H2H
        if 'h2h_home_wins' in features and 'h2h_total' in features:
            total = features['h2h_total']
            if total > 0:
                win_rate = features['h2h_home_wins'] / total * 100
                if win_rate > 60:
                    points.append(f"• Historique favorable ({win_rate:.0f}% de victoires)")
                elif win_rate < 30:
                    points.append(f"• Historique défavorable ({win_rate:.0f}% de victoires)")
        
        # Analyse value bet
        if 'value_bet_info' in prediction:
            vb = prediction['value_bet_info']
            if vb['is_value_bet']:
                points.append(f"• VALUE BET : divergence de {vb['divergence_percent']:.1f}%")
                points.append(f"• Expected Value : +{vb['expected_value']*100:.1f}%")
        
        if not points:
            points.append("• Analyse standard - aucun signal particulier")
        
        return points
    
    def interpret_prediction(
        self,
        match_data: Dict,
        model_prediction: Dict,
        features: Optional[Dict] = None
    ) -> Dict:
        """
        Interprète une prédiction complète.
        
        Args:
            match_data: Données du match (équipes, date, ligue, cotes)
            model_prediction: Sortie du modèle (probas, prédiction)
            features: Features additionnelles pour l'analyse
            
        Returns:
            Dictionnaire d'interprétation complet
        """
        # Extraire les probabilités
        probs = model_prediction.get('probabilities', {})
        predicted_class = model_prediction.get('predicted_class', 2)
        
        # Mapping des classes
        class_mapping = {2: 'home', 1: 'draw', 0: 'away'}
        outcome_labels = {2: 'Victoire à domicile', 1: 'Match nul', 0: 'Victoire à l\'extérieur'}
        
        predicted_outcome = class_mapping.get(predicted_class, 'unknown')
        model_prob = probs.get(predicted_outcome, 0)
        
        # Niveau de confiance
        confidence_level, confidence_label = self.get_confidence_level(model_prob)
        
        # Analyse value bet
        odd_key = f"B365{predicted_outcome.upper()[0]}"
        bookmaker_odd = match_data.get(odd_key, match_data.get('B365H', 2.0))
        value_bet_info = self.detect_value_bet(model_prob, bookmaker_odd)
        
        # Points d'analyse
        features = features or {}
        analysis_points = self.generate_analysis_points(features, {
            'value_bet_info': value_bet_info
        })
        
        # Construire le résultat
        result = {
            'match_info': {
                'home_team': match_data.get('HomeTeam', 'Unknown'),
                'away_team': match_data.get('AwayTeam', 'Unknown'),
                'date': match_data.get('Date', ''),
                'league': match_data.get('league', 'Unknown')
            },
            'prediction': {
                'outcome': predicted_outcome,
                'outcome_label': outcome_labels.get(predicted_class, 'Inconnu'),
                'probabilities': probs,
                'confidence_level': confidence_level,
                'confidence_label': confidence_label,
                'model_probability': model_prob
            },
            'value_bet': value_bet_info,
            'analysis': {
                'points': analysis_points,
                'features_used': list(features.keys()) if features else []
            },
            'recommendation': self._get_recommendation(confidence_level, value_bet_info)
        }
        
        # Historique
        self.interpretation_history.append(result)
        
        return result
    
    def _get_recommendation(
        self,
        confidence_level: str,
        value_bet_info: Dict
    ) -> Dict:
        """
        Génère une recommandation de mise.
        
        Args:
            confidence_level: Niveau de confiance (low/medium/high)
            value_bet_info: Info sur le value bet
            
        Returns:
            Dictionnaire de recommandation
        """
        kelly_criterion = 0.0
        
        if value_bet_info['is_value_bet'] and value_bet_info['expected_value'] > 0:
            # Formule de Kelly (fractionnée à 25% pour réduire le risque)
            kelly_criterion = (
                value_bet_info['expected_value'] / 
                (value_bet_info['implied_probability'] * (1 / value_bet_info['implied_probability'] - 1))
            ) * 0.25
            kelly_criterion = max(0, min(kelly_criterion, 0.25))  # Cap à 25%
        
        recommendation_map = {
            'high': {
                'action': 'MISE RECOMMANDÉE',
                'stake_percent': min(kelly_criterion * 100, 5) if kelly_criterion > 0 else 2,
                'reasoning': 'Confiance élevée du modèle'
            },
            'medium': {
                'action': 'MISE OPTIONNELLE',
                'stake_percent': min(kelly_criterion * 100, 2) if kelly_criterion > 0 else 1,
                'reasoning': 'Confiance moyenne - réduire la mise'
            },
            'low': {
                'action': 'NE PAS MISER',
                'stake_percent': 0,
                'reasoning': 'Confiance insuffisante'
            }
        }
        
        base_rec = recommendation_map.get(confidence_level, recommendation_map['low'])
        
        # Bonus pour value bet
        if value_bet_info['is_value_bet'] and confidence_level != 'low':
            base_rec['action'] = 'VALUE BET - MISE RECOMMANDÉE'
            base_rec['reasoning'] += ' + Value bet détecté'
        
        return {
            **base_rec,
            'kelly_criterion': kelly_criterion,
            'value_bet_detected': value_bet_info['is_value_bet']
        }
    
    def generate_report(
        self,
        interpretation: Dict,
        output_format: str = 'text'
    ) -> str:
        """
        Génère un rapport depuis une interprétation.
        
        Args:
            interpretation: Résultat de interpret_prediction()
            output_format: 'text' ou 'json'
            
        Returns:
            Rapport formaté
        """
        if output_format == 'json':
            return json.dumps(interpretation, indent=2, default=str)
        
        # Format texte avec template
        confidence = interpretation['prediction']['confidence_level']
        template_key = f"{confidence}_confidence"
        
        if confidence == 'high' and interpretation['prediction']['outcome'] == 'home':
            template_key = 'high_confidence_home'
        
        template = self.REPORT_TEMPLATES.get(template_key, self.REPORT_TEMPLATES['medium_confidence'])
        
        # Remplir le template
        match_info = interpretation['match_info']
        pred = interpretation['prediction']
        vb = interpretation['value_bet']
        
        report = template.format(
            home_team=match_info['home_team'],
            away_team=match_info['away_team'],
            match_date=match_info['date'],
            league=match_info['league'],
            predicted_outcome=pred['outcome_label'],
            confidence=pred['model_probability'] * 100,
            model_prob=pred['model_probability'] * 100,
            bookmaker_odd=1 / vb['implied_probability'] if vb['implied_probability'] > 0 else 0,
            implied_prob=vb['implied_probability'] * 100,
            divergence=vb['divergence_percent'],
            value_bet_status=vb['recommendation'],
            analysis_points='\n'.join(interpretation['analysis']['points'])
        )
        
        return report.strip()
    
    def batch_interpret(
        self,
        predictions: List[Dict],
        match_datas: List[Dict],
        features_list: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Interprète un batch de prédictions.
        
        Args:
            predictions: Liste des prédictions du modèle
            match_datas: Liste des données de matchs
            features_list: Liste des features par match
            
        Returns:
            Liste des interprétations
        """
        results = []
        features_list = features_list or [{} for _ in predictions]
        
        for pred, match_data, features in zip(predictions, match_datas, features_list):
            interpretation = self.interpret_prediction(match_data, pred, features)
            results.append(interpretation)
        
        return results
    
    def get_summary_statistics(self) -> Dict:
        """
        Retourne des statistiques sur les interprétations passées.
        
        Returns:
            Dictionnaire de statistiques
        """
        if not self.interpretation_history:
            return {'total': 0}
        
        total = len(self.interpretation_history)
        confidence_counts = {}
        value_bets_count = 0
        
        for interp in self.interpretation_history:
            level = interp['prediction']['confidence_level']
            confidence_counts[level] = confidence_counts.get(level, 0) + 1
            
            if interp['value_bet']['is_value_bet']:
                value_bets_count += 1
        
        return {
            'total_interpretations': total,
            'confidence_distribution': confidence_counts,
            'value_bets_detected': value_bets_count,
            'value_bet_rate': value_bets_count / total if total > 0 else 0
        }
