"""
Polymarket Gamma API Client pour l'extraction de probabilités de marchés football.
API publique sans authentification - https://gamma-api.polymarket.com
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class MarketOutcome:
    """Représente un outcome d'un marché Polymarket."""
    name: str
    price: float  # Probabilité implicite (0-1)
    volume: float = 0.0
    open_interest: float = 0.0


@dataclass
class FootballMarket:
    """Représente un marché football sur Polymarket."""
    event_id: str
    title: str
    outcomes: List[MarketOutcome] = field(default_factory=list)
    volume: float = 0.0
    liquidity: float = 0.0
    close_date: Optional[datetime] = None
    category: str = ""
    url: str = ""


@dataclass
class OrderbookLevel:
    """Niveau de carnet d'ordres."""
    price: float
    size: float
    count: int = 0


@dataclass
class OrderbookSnapshot:
    """Snapshot du carnet d'ordres pour un outcome."""
    outcome: str
    bids: List[OrderbookLevel] = field(default_factory=list)
    asks: List[OrderbookLevel] = field(default_factory=list)
    spread: float = 0.0
    mid_price: float = 0.0


class RateLimiter:
    """Gestionnaire de rate limiting pour respecter ~1 req/sec."""
    
    def __init__(self, calls_per_second: float = 1.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0
    
    def wait(self):
        """Attend si nécessaire pour respecter le rate limit."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_call_time = time.time()


class PolymarketClient:
    """
    Client pour l'API Gamma de Polymarket.
    Permet de rechercher et extraire les probabilités des marchés football.
    
    API Docs: https://gamma-api.polymarket.com
    """
    
    BASE_URL = "https://gamma-api.polymarket.com"
    
    def __init__(self, rate_limit: float = 1.0, timeout: int = 10):
        """
        Initialise le client Polymarket.
        
        Args:
            rate_limit: Nombre de requêtes par seconde (défaut: 1.0)
            timeout: Timeout des requêtes en secondes
        """
        self.rate_limiter = RateLimiter(rate_limit)
        self.timeout = timeout
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Crée une session HTTP avec retry automatique."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Effectue une requête GET avec gestion des erreurs et rate limiting.
        
        Args:
            endpoint: Endpoint API (sans le BASE_URL)
            params: Paramètres de requête
            
        Returns:
            Réponse JSON ou None en cas d'échec
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        self.rate_limiter.wait()
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error {e.response.status_code}: {endpoint}")
            if e.response.status_code == 429:
                logger.warning("Rate limit exceeded, waiting 2 seconds...")
                time.sleep(2)
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            return None
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def search_football_markets(
        self,
        keywords: Optional[List[str]] = None,
        category: str = "sports",
        limit: int = 50,
        closed: bool = False
    ) -> List[FootballMarket]:
        """
        Recherche des marchés football via l'API Gamma.
        
        Args:
            keywords: Mots-clés pour filtrer (ex: ["football", "soccer", "premier league"])
            category: Catégorie de marché (défaut: "sports")
            limit: Nombre maximum de résultats
            closed: Inclure les marchés fermés
            
        Returns:
            Liste de FootballMarket
        """
        default_keywords = ["football", "soccer", "premier league", "champions league", 
                          "la liga", "serie a", "bundesliga", "ligue 1"]
        search_keywords = keywords or default_keywords
        
        markets = []
        
        # Endpoint pour lister les marchés
        endpoint = "/events"
        params = {
            "category": category,
            "limit": limit,
            "closed": str(closed).lower()
        }
        
        data = self._request(endpoint, params)
        if not data:
            return markets
        
        events = data.get("events", []) if isinstance(data, dict) else []
        
        for event in events:
            event_id = event.get("id", "")
            title = event.get("title", "").lower()
            
            # Filtrage par keywords
            if not any(kw.lower() in title for kw in search_keywords):
                continue
            
            # Récupérer les détails du marché
            market_data = self._get_market_details(event_id)
            if market_data:
                market = self._parse_market(event_id, market_data)
                if market:
                    markets.append(market)
        
        logger.info(f"Found {len(markets)} football markets matching keywords")
        return markets
    
    def _get_market_details(self, event_id: str) -> Optional[Dict]:
        """Récupère les détails d'un marché spécifique."""
        endpoint = f"/events/{event_id}"
        return self._request(endpoint)
    
    def _parse_market(self, event_id: str, data: Dict) -> Optional[FootballMarket]:
        """Parse les données brutes en objet FootballMarket."""
        try:
            title = data.get("title", "")
            volume = float(data.get("volume", 0))
            liquidity = float(data.get("liquidity", 0))
            category = data.get("category", "")
            
            # Date de clôture
            close_date = None
            close_ts = data.get("closeDate") or data.get("end_date")
            if close_ts:
                try:
                    close_date = datetime.fromtimestamp(int(close_ts) / 1000)
                except (ValueError, TypeError):
                    pass
            
            # Parser les outcomes
            outcomes = []
            outcome_prices = data.get("outcomes", [])
            
            for outcome in outcome_prices:
                name = outcome.get("outcome", outcome.get("name", ""))
                price = float(outcome.get("price", outcome.get("outcomePrice", 0)))
                vol = float(outcome.get("volume", 0))
                oi = float(outcome.get("openInterest", 0))
                
                outcomes.append(MarketOutcome(
                    name=name,
                    price=price,
                    volume=vol,
                    open_interest=oi
                ))
            
            url = f"https://polymarket.com/event/{event_id}"
            
            return FootballMarket(
                event_id=event_id,
                title=title,
                outcomes=outcomes,
                volume=volume,
                liquidity=liquidity,
                close_date=close_date,
                category=category,
                url=url
            )
        except Exception as e:
            logger.error(f"Error parsing market {event_id}: {e}")
            return None
    
    def extract_match_odds(self, market: FootballMarket) -> Dict[str, Any]:
        """
        Extrait les probabilités et cotes implicites depuis un marché.
        
        Args:
            market: Objet FootballMarket
            
        Returns:
            Dict avec probabilités, cotes décimales, et overround
        """
        if not market.outcomes:
            return {"error": "No outcomes available"}
        
        probabilities = {}
        decimal_odds = {}
        total_probability = 0.0
        
        for outcome in market.outcomes:
            prob = outcome.price  # Déjà en format 0-1 sur Polymarket
            probabilities[outcome.name] = prob
            total_probability += prob
            
            # Conversion en cote décimale (1/probabilité)
            if prob > 0:
                decimal_odds[outcome.name] = round(1.0 / prob, 3)
            else:
                decimal_odds[outcome.name] = float('inf')
        
        # Calcul de l'overround (marge du marché)
        overround = (total_probability - 1.0) * 100 if total_probability > 0 else 0
        
        return {
            "event_id": market.event_id,
            "title": market.title,
            "probabilities": probabilities,
            "decimal_odds": decimal_odds,
            "total_probability": total_probability,
            "overround_percent": round(overround, 2),
            "volume": market.volume,
            "liquidity": market.liquidity,
            "close_date": market.close_date.isoformat() if market.close_date else None
        }
    
    def get_orderbook_snapshot(self, event_id: str) -> List[OrderbookSnapshot]:
        """
        Récupère un snapshot du carnet d'ordres pour un marché.
        
        Args:
            event_id: ID de l'événement Polymarket
            
        Returns:
            Liste de OrderbookSnapshot par outcome
        """
        endpoint = f"/events/{event_id}/orderbook"
        data = self._request(endpoint)
        
        if not data:
            logger.warning(f"No orderbook data for event {event_id}")
            return []
        
        snapshots = []
        orderbook_data = data.get("orderbook", {}) or data
        
        for outcome_name, levels in orderbook_data.items():
            if not isinstance(levels, dict):
                continue
            
            bids = []
            asks = []
            
            # Parser les bids (achats)
            for bid in levels.get("bids", []):
                bids.append(OrderbookLevel(
                    price=float(bid.get("price", 0)),
                    size=float(bid.get("size", 0)),
                    count=int(bid.get("count", 0))
                ))
            
            # Parser les asks (ventes)
            for ask in levels.get("asks", []):
                asks.append(OrderbookLevel(
                    price=float(ask.get("price", 0)),
                    size=float(ask.get("size", 0)),
                    count=int(ask.get("count", 0))
                ))
            
            # Calculer spread et mid price
            spread = 0.0
            mid_price = 0.0
            
            if bids and asks:
                best_bid = max(b.price for b in bids)
                best_ask = min(a.price for a in asks)
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2
            
            snapshots.append(OrderbookSnapshot(
                outcome=outcome_name,
                bids=sorted(bids, key=lambda x: x.price, reverse=True),
                asks=sorted(asks, key=lambda x: x.price),
                spread=round(spread, 4),
                mid_price=round(mid_price, 4)
            ))
        
        return snapshots
    
    def get_market_by_id(self, event_id: str) -> Optional[FootballMarket]:
        """
        Récupère un marché spécifique par son ID.
        
        Args:
            event_id: ID de l'événement
            
        Returns:
            FootballMarket ou None
        """
        data = self._get_market_details(event_id)
        if data:
            return self._parse_market(event_id, data)
        return None
    
    def get_active_markets_count(self, category: str = "sports") -> int:
        """
        Compte le nombre de marchés actifs dans une catégorie.
        
        Args:
            category: Catégorie à compter
            
        Returns:
            Nombre de marchés actifs
        """
        endpoint = "/events"
        params = {
            "category": category,
            "closed": "false",
            "limit": 1
        }
        
        data = self._request(endpoint, params)
        if not data:
            return 0
        
        # L'API peut retourner un total dans les métadonnées
        total = data.get("total", 0)
        if total:
            return total
        
        # Sinon, compter manuellement (approximation)
        return len(data.get("events", []))


class PolymarketHistorical:
    """
    Client pour récupérer l'historique des prix Polymarket.
    Utile pour le backtesting des stratégies.
    """
    
    BASE_URL = "https://gamma-api.polymarket.com"
    
    def __init__(self, rate_limit: float = 0.5, timeout: int = 15):
        """
        Initialise le client historique.
        
        Args:
            rate_limit: Requêtes par seconde (plus bas pour l'historique)
            timeout: Timeout en secondes
        """
        self.rate_limiter = RateLimiter(rate_limit)
        self.timeout = timeout
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Crée une session HTTP avec retry."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Requête GET avec rate limiting."""
        url = f"{self.BASE_URL}{endpoint}"
        
        self.rate_limiter.wait()
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def get_price_history(
        self,
        event_id: str,
        outcome: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resolution: str = "1h"
    ) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des prix pour un marché.
        
        Args:
            event_id: ID de l'événement
            outcome: Nom de l'outcome spécifique (optionnel)
            start_date: Date de début (défaut: 30 jours avant)
            end_date: Date de fin (défaut: maintenant)
            resolution: Résolution temporelle (1m, 5m, 15m, 1h, 1d)
            
        Returns:
            Liste de points de données {timestamp, price, volume}
        """
        endpoint = f"/events/{event_id}/history"
        
        params = {"resolution": resolution}
        
        if start_date:
            params["start"] = int(start_date.timestamp() * 1000)
        else:
            # Défaut: 30 jours avant
            default_start = datetime.now() - timedelta(days=30)
            params["start"] = int(default_start.timestamp() * 1000)
        
        if end_date:
            params["end"] = int(end_date.timestamp() * 1000)
        
        if outcome:
            params["outcome"] = outcome
        
        data = self._request(endpoint, params)
        
        if not data:
            logger.warning(f"No history data for event {event_id}")
            return []
        
        history = data.get("history", []) or data.get("data", [])
        
        # Normaliser le format
        normalized = []
        for point in history:
            normalized.append({
                "timestamp": point.get("timestamp", point.get("time", 0)),
                "price": float(point.get("price", point.get("outcomePrice", 0))),
                "volume": float(point.get("volume", 0)),
                "outcome": point.get("outcome", outcome)
            })
        
        logger.info(f"Retrieved {len(normalized)} history points for {event_id}")
        return normalized
    
    def get_ohlcv(
        self,
        event_id: str,
        outcome: str,
        timeframe: str = "1h",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Récupère des données OHLCV (Open, High, Low, Close, Volume).
        
        Args:
            event_id: ID de l'événement
            outcome: Nom de l'outcome
            timeframe: Unité de temps (1m, 5m, 15m, 1h, 1d)
            limit: Nombre de bougies
            
        Returns:
            Liste de bougies OHLCV
        """
        endpoint = f"/events/{event_id}/candles"
        
        params = {
            "outcome": outcome,
            "timeframe": timeframe,
            "limit": limit
        }
        
        data = self._request(endpoint, params)
        
        if not data:
            return []
        
        candles = data.get("candles", []) or data.get("data", [])
        
        normalized = []
        for candle in candles:
            normalized.append({
                "timestamp": candle.get("timestamp", candle.get("time", 0)),
                "open": float(candle.get("open", 0)),
                "high": float(candle.get("high", 0)),
                "low": float(candle.get("low", 0)),
                "close": float(candle.get("close", 0)),
                "volume": float(candle.get("volume", 0))
            })
        
        return normalized
    
    def calculate_realized_probability(
        self,
        event_id: str,
        outcome: str,
        lookback_days: int = 7
    ) -> Dict[str, float]:
        """
        Calcule la probabilité réalisée moyenne sur une période.
        Utile pour évaluer la précision des marchés.
        
        Args:
            event_id: ID de l'événement
            outcome: Nom de l'outcome
            lookback_days: Nombre de jours à analyser
            
        Returns:
            Stats de probabilité (moyenne, min, max, std)
        """
        start_date = datetime.now() - timedelta(days=lookback_days)
        history = self.get_price_history(event_id, outcome, start_date)
        
        if not history:
            return {"error": "No data available"}
        
        prices = [p["price"] for p in history]
        
        if not prices:
            return {"error": "No prices found"}
        
        import statistics
        
        return {
            "mean": statistics.mean(prices),
            "median": statistics.median(prices),
            "min": min(prices),
            "max": max(prices),
            "std": statistics.stdev(prices) if len(prices) > 1 else 0,
            "current": prices[-1] if prices else 0,
            "data_points": len(prices)
        }


# Exemple d'utilisation
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialiser le client
    client = PolymarketClient(rate_limit=1.0)
    
    # Rechercher des marchés football
    print("Recherche de marchés football...")
    markets = client.search_football_markets(
        keywords=["premier league", "champions league"],
        limit=10
    )
    
    for market in markets[:3]:  # Afficher les 3 premiers
        print(f"\nMarché: {market.title}")
        print(f"Volume: ${market.volume:,.0f}")
        print(f"Liquidité: ${market.liquidity:,.0f}")
        
        # Extraire les probabilités
        odds = client.extract_match_odds(market)
        if "error" not in odds:
            print(f"Probabilités: {odds['probabilities']}")
            print(f"Cotes décimales: {odds['decimal_odds']}")
            print(f"Overround: {odds['overround_percent']}%")
    
    # Récupérer un carnet d'ordres
    if markets:
        print("\n\nRécupération du carnet d'ordres...")
        orderbook = client.get_orderbook_snapshot(markets[0].event_id)
        for snapshot in orderbook:
            print(f"\nOutcome: {snapshot.outcome}")
            print(f"Spread: {snapshot.spread:.4f}")
            print(f"Mid Price: {snapshot.mid_price:.4f}")
            if snapshot.bids:
                print(f"Meilleur bid: {snapshot.bids[0].price} ({snapshot.bids[0].size})")
            if snapshot.asks:
                print(f"Meilleur ask: {snapshot.asks[0].price} ({snapshot.asks[0].size})")
    
    # Historique des prix
    print("\n\n--- Historique des prix ---")
    historical = PolymarketHistorical(rate_limit=0.5)
    
    if markets:
        history = historical.get_price_history(
            markets[0].event_id,
            resolution="1h"
        )
        print(f"Points d'historique récupérés: {len(history)}")
        if history:
            print(f"Dernier prix: {history[-1]['price']:.4f}")
