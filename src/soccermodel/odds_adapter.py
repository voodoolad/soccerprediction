"""
Odds Adapter - Load and process market odds with vig removal
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import json

@dataclass
class OddsSnapshot:
    market: str
    prices: Dict[str, float]  # Already vig-adjusted if apply_vig_removal=True
    raw_prices: Dict[str, float]  # Original prices before vig removal
    timestamp: str
    vig_removed: bool = False

def remove_vig_proportional(prices: Dict[str, float]) -> Dict[str, float]:
    """
    Remove vig using proportional (fair odds) method
    
    Args:
        prices: Dict of outcome -> decimal odds
    
    Returns:
        Dict of outcome -> vig-adjusted decimal odds
    
    Example:
        >>> remove_vig_proportional({'home': 2.10, 'draw': 3.40, 'away': 3.60})
        {'home': 2.21, 'draw': 3.58, 'away': 3.79}
    """
    if not prices:
        return prices
    
    # Calculate implied probabilities
    implied_probs = {outcome: 1.0 / odds for outcome, odds in prices.items()}
    
    # Total implied probability (overround)
    total_prob = sum(implied_probs.values())
    
    # If no overround, return original prices
    if total_prob <= 1.0:
        return prices.copy()
    
    # Remove vig proportionally
    fair_probs = {outcome: prob / total_prob for outcome, prob in implied_probs.items()}
    
    # Convert back to odds
    fair_odds = {outcome: 1.0 / prob for outcome, prob in fair_probs.items()}
    
    return fair_odds

def remove_vig_shin(prices: Dict[str, float], max_iter: int = 100, tolerance: float = 1e-6) -> Dict[str, float]:
    """
    Remove vig using Shin's method (accounts for insider trading/informed money)
    More sophisticated than proportional, better for markets with strong favorites
    
    Args:
        prices: Dict of outcome -> decimal odds
        max_iter: Maximum iterations for convergence
        tolerance: Convergence tolerance
    
    Returns:
        Dict of outcome -> vig-adjusted decimal odds
    """
    import numpy as np
    
    if not prices or len(prices) < 2:
        return prices
    
    # Convert to implied probabilities
    implied = np.array([1.0 / odds for odds in prices.values()])
    outcomes = list(prices.keys())
    
    # Total overround
    total = implied.sum()
    
    if total <= 1.0:
        return prices.copy()
    
    # Shin's method: solve for z (insider trading parameter)
    # Uses iterative approach
    z = 0.0
    
    for _ in range(max_iter):
        # Calculate fair probabilities given current z
        numerator = np.sqrt(z**2 + 4 * (1 - z) * implied / total) - z
        denominator = 2 * (1 - z)
        fair_probs = numerator / denominator
        
        # Check convergence
        if abs(fair_probs.sum() - 1.0) < tolerance:
            break
        
        # Update z
        z = (total - 1) / (len(implied) - 1)
        z = max(0, min(z, 0.3))  # Cap z at reasonable bounds
    
    # Normalize to ensure probabilities sum to 1
    fair_probs = fair_probs / fair_probs.sum()
    
    # Convert back to odds
    fair_odds = {outcome: 1.0 / prob for outcome, prob in zip(outcomes, fair_probs)}
    
    return fair_odds

def load_odds_from_json(path: str, apply_vig_removal: bool = True, method: str = 'proportional') -> Dict[str, OddsSnapshot]:
    """
    Load odds from JSON file with optional vig removal
    
    Args:
        path: Path to JSON file
        apply_vig_removal: Whether to remove vig from prices
        method: 'proportional' or 'shin'
    
    Returns:
        Dict of market_name -> OddsSnapshot
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    
    out = {}
    markets = payload.get("markets", {})
    
    for mkt, data in markets.items():
        ts = data.get("ts") or data.get("timestamp") or ""
        
        # Extract raw prices (exclude metadata)
        raw_prices = {k.lower(): float(v) for k, v in data.items() 
                      if k not in ("ts", "timestamp") and isinstance(v, (int, float))}
        
        # Apply vig removal if requested
        if apply_vig_removal and raw_prices:
            if method == 'shin':
                adjusted_prices = remove_vig_shin(raw_prices)
            else:
                adjusted_prices = remove_vig_proportional(raw_prices)
        else:
            adjusted_prices = raw_prices.copy()
        
        out[mkt] = OddsSnapshot(
            market=mkt, 
            prices=adjusted_prices,
            raw_prices=raw_prices,
            timestamp=ts,
            vig_removed=apply_vig_removal
        )
    
    return out

def kelly_fraction(p: float, o: float) -> float:
    """
    Calculate Kelly fraction for a bet
    
    Args:
        p: True probability of winning
        o: Decimal odds offered
    
    Returns:
        Kelly fraction (0 if no edge, negative if bad bet)
    """
    if o <= 1.0:
        return 0.0
    
    f = (p * o - 1.0) / (o - 1.0)
    return max(0.0, f)

def american_from_decimal(d: float) -> int:
    """Convert decimal odds to American format"""
    if d >= 2.0:
        return int(round((d - 1.0) * 100))
    else:
        return int(round(-100.0 / (d - 1.0)))

def calculate_overround(prices: Dict[str, float]) -> float:
    """
    Calculate market overround (vig percentage)
    
    Args:
        prices: Dict of outcome -> decimal odds
    
    Returns:
        Overround as percentage (e.g., 5.2 for 5.2% vig)
    """
    if not prices:
        return 0.0
    
    implied_total = sum(1.0 / odds for odds in prices.values())
    overround = (implied_total - 1.0) * 100
    
    return max(0.0, overround)

def find_best_price(market_prices: Dict[str, Dict[str, float]], outcome: str) -> tuple:
    """
    Find best available price across multiple books
    
    Args:
        market_prices: Dict of book_name -> {outcome -> odds}
        outcome: Outcome to find best price for
    
    Returns:
        (best_odds, book_name) or (None, None) if not found
    """
    best_odds = None
    best_book = None
    
    for book, prices in market_prices.items():
        outcome_lower = outcome.lower()
        if outcome_lower in prices:
            odds = prices[outcome_lower]
            if best_odds is None or odds > best_odds:
                best_odds = odds
                best_book = book
    
    return best_odds, best_book

def is_materially_better(price: float, consensus: float, threshold_decimal: float = 0.03) -> bool:
    """
    Check if a price is materially better than consensus
    
    Args:
        price: Offered price (decimal odds)
        consensus: Consensus/fair price (decimal odds)
        threshold_decimal: Minimum decimal odds difference (default 0.03)
    
    Returns:
        True if price is materially better
    """
    if price <= 0 or consensus <= 0:
        return False
    
    # Check decimal difference
    if price - consensus >= threshold_decimal:
        return True
    
    # Check implied probability difference (at least 2 percentage points)
    price_implied = 1.0 / price
    consensus_implied = 1.0 / consensus
    prob_diff = consensus_implied - price_implied
    
    if prob_diff >= 0.02:  # 2 percentage points
        return True
    
    return False
