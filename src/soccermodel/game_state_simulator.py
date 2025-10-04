"""
Game State Simulator - Model how teams behave in different match states
Handles: trailing/leading scenarios, time bands, red cards, late substitutions
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
from enum import Enum

class GameState(Enum):
    """Match state categories"""
    LEVEL = "level"
    TRAILING_1 = "trailing_1"
    TRAILING_2_PLUS = "trailing_2plus"
    LEADING_1 = "leading_1"
    LEADING_2_PLUS = "leading_2plus"
    RED_CARD_FOR = "red_card_for"  # Team has 10 players
    RED_CARD_AGAINST = "red_card_against"  # Opponent has 10 players

@dataclass
class TimeBand:
    """Time period in match"""
    start: int
    end: int
    
    def contains(self, minute: int) -> bool:
        return self.start <= minute < self.end

# Standard time bands
TIME_BANDS = [
    TimeBand(0, 15),
    TimeBand(15, 30),
    TimeBand(30, 46),  # Includes first-half stoppage
    TimeBand(46, 60),
    TimeBand(60, 75),
    TimeBand(75, 96)   # Includes second-half stoppage
]

@dataclass
class StateMultipliers:
    """Multipliers for team behavior in different states"""
    shots: float = 1.0
    sot: float = 1.0
    xg_rate: float = 1.0
    corners: float = 1.0
    fouls: float = 1.0
    cards: float = 1.0
    possession: float = 1.0

# Default multipliers based on typical team behavior patterns
DEFAULT_MULTIPLIERS = {
    GameState.LEVEL: StateMultipliers(),
    
    # Trailing teams become more aggressive
    GameState.TRAILING_1: StateMultipliers(
        shots=1.25, sot=1.20, xg_rate=1.15, corners=1.15, fouls=1.10, cards=1.15
    ),
    GameState.TRAILING_2_PLUS: StateMultipliers(
        shots=1.40, sot=1.30, xg_rate=1.20, corners=1.25, fouls=1.15, cards=1.20
    ),
    
    # Leading teams become more defensive
    GameState.LEADING_1: StateMultipliers(
        shots=0.85, sot=0.90, xg_rate=0.90, corners=0.90, fouls=0.95, cards=1.10
    ),
    GameState.LEADING_2_PLUS: StateMultipliers(
        shots=0.70, sot=0.80, xg_rate=0.85, corners=0.80, fouls=0.90, cards=1.05
    ),
    
    # Red card scenarios
    GameState.RED_CARD_FOR: StateMultipliers(
        shots=0.75, sot=0.70, xg_rate=0.70, corners=0.80, fouls=1.20, cards=1.30
    ),
    GameState.RED_CARD_AGAINST: StateMultipliers(
        shots=1.30, sot=1.25, xg_rate=1.35, corners=1.20, fouls=0.90, cards=0.95
    )
}

class GameStateSimulator:
    """
    Simulate match progression accounting for game states and time bands
    """
    
    def __init__(self, home_base_lambda: float, away_base_lambda: float,
                 home_style: str = 'balanced', away_style: str = 'balanced'):
        """
        Args:
            home_base_lambda: Home team's base expected goals per 90
            away_base_lambda: Away team's base expected goals per 90
            home_style: 'attacking', 'balanced', 'defensive'
            away_style: 'attacking', 'balanced', 'defensive'
        """
        self.home_lambda = home_base_lambda
        self.away_lambda = away_base_lambda
        self.home_style = home_style
        self.away_style = away_style
        
        # Style adjustments for state multipliers
        self.style_factors = {
            'attacking': 1.15,    # More extreme swings
            'balanced': 1.0,
            'defensive': 0.85     # Less aggressive when trailing
        }
    
    def simulate_match(self, n_sims: int = 10000, include_states: bool = True) -> Dict:
        """
        Run Monte Carlo simulation of match
        
        Args:
            n_sims: Number of simulations
            include_states: Whether to model game state effects
        
        Returns:
            Dict with scoreline probabilities and derived markets
        """
        scorelines = []
        goal_times_home = []
        goal_times_away = []
        
        for _ in range(n_sims):
            score, home_times, away_times = self._simulate_single_match(include_states)
            scorelines.append(score)
            goal_times_home.extend(home_times)
            goal_times_away.extend(away_times)
        
        # Calculate scoreline PMF
        pmf = {}
        for score in scorelines:
            pmf[score] = pmf.get(score, 0) + 1 / n_sims
        
        # Market probabilities
        markets = self._calculate_markets(pmf)
        
        # Time-based probabilities
        markets['late_goals'] = self._calculate_late_goal_probs(goal_times_home, goal_times_away)
        
        return {
            'scoreline_pmf': pmf,
            'markets': markets,
            'goal_times_home': goal_times_home,
            'goal_times_away': goal_times_away
        }
    
    def _simulate_single_match(self, include_states: bool) -> Tuple[Tuple[int, int], List[int], List[int]]:
        """Simulate one match minute-by-minute"""
        home_score = 0
        away_score = 0
        home_goal_times = []
        away_goal_times = []
        
        # Player counts (for red card scenarios)
        home_players = 11
        away_players = 11
        
        for minute in range(1, 96):  # Include stoppage time
            # Determine current state
            state_home, state_away = self._get_game_states(
                home_score, away_score, home_players, away_players, minute
            )
            
            # Get multipliers for this state and time
            if include_states:
                mult_home = self._get_multiplier(state_home, minute, self.home_style)
                mult_away = self._get_multiplier(state_away, minute, self.away_style)
            else:
                mult_home = StateMultipliers()
                mult_away = StateMultipliers()
            
            # Calculate goal probability for this minute
            # Base rate per 90 minutes -> per minute
            home_lambda_minute = (self.home_lambda / 90) * mult_home.xg_rate
            away_lambda_minute = (self.away_lambda / 90) * mult_away.xg_rate
            
            # Simulate goals (Poisson events)
            if np.random.random() < home_lambda_minute:
                home_score += 1
                home_goal_times.append(minute)
            
            if np.random.random() < away_lambda_minute:
                away_score += 1
                away_goal_times.append(minute)
            
            # Simulate red cards (low probability)
            if np.random.random() < 0.0003:  # ~0.03% per minute = ~3% per game
                if np.random.random() < 0.5:
                    home_players = 10
                else:
                    away_players = 10
        
        return (home_score, away_score), home_goal_times, away_goal_times
    
    def _get_game_states(self, home_score: int, away_score: int, 
                        home_players: int, away_players: int, 
                        minute: int) -> Tuple[GameState, GameState]:
        """Determine game state for both teams"""
        diff = home_score - away_score
        
        # Red cards override other states
        if home_players < 11:
            state_home = GameState.RED_CARD_FOR
        elif away_players < 11:
            state_home = GameState.RED_CARD_AGAINST
        # Score-based states (more important in late game)
        elif diff == 0:
            state_home = GameState.LEVEL
        elif diff == 1:
            state_home = GameState.LEADING_1 if minute > 60 else GameState.LEVEL
        elif diff >= 2:
            state_home = GameState.LEADING_2_PLUS
        elif diff == -1:
            state_home = GameState.TRAILING_1 if minute > 60 else GameState.LEVEL
        else:  # diff <= -2
            state_home = GameState.TRAILING_2_PLUS
        
        # Away state is mirror of home
        if away_players < 11:
            state_away = GameState.RED_CARD_FOR
        elif home_players < 11:
            state_away = GameState.RED_CARD_AGAINST
        elif diff == 0:
            state_away = GameState.LEVEL
        elif diff == -1:
            state_away = GameState.LEADING_1 if minute > 60 else GameState.LEVEL
        elif diff <= -2:
            state_away = GameState.LEADING_2_PLUS
        elif diff == 1:
            state_away = GameState.TRAILING_1 if minute > 60 else GameState.LEVEL
        else:  # diff >= 2
            state_away = GameState.TRAILING_2_PLUS
        
        return state_home, state_away
    
    def _get_multiplier(self, state: GameState, minute: int, style: str) -> StateMultipliers:
        """Get state multipliers adjusted for team style and time"""
        base = DEFAULT_MULTIPLIERS[state]
        style_factor = self.style_factors[style]
        
        # Time factor: states matter more in late game
        if minute > 75:
            time_factor = 1.2
        elif minute > 60:
            time_factor = 1.1
        else:
            time_factor = 1.0
        
        # Apply style and time adjustments
        return StateMultipliers(
            shots=1 + (base.shots - 1) * style_factor * time_factor,
            sot=1 + (base.sot - 1) * style_factor * time_factor,
            xg_rate=1 + (base.xg_rate - 1) * style_factor * time_factor,
            corners=1 + (base.corners - 1) * time_factor,
            fouls=1 + (base.fouls - 1) * time_factor,
            cards=1 + (base.cards - 1) * time_factor,
            possession=base.possession
        )
    
    def _calculate_markets(self, pmf: Dict[Tuple[int, int], float]) -> Dict[str, float]:
        """Calculate market probabilities from scoreline PMF"""
        markets = {}
        
        # 1X2
        markets['home_win'] = sum(p for (h, a), p in pmf.items() if h > a)
        markets['draw'] = sum(p for (h, a), p in pmf.items() if h == a)
        markets['away_win'] = sum(p for (h, a), p in pmf.items() if h < a)
        
        # Totals
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            markets[f'over_{line}'] = sum(p for (h, a), p in pmf.items() if h + a > line)
        
        # BTTS
        markets['btts_yes'] = sum(p for (h, a), p in pmf.items() if h > 0 and a > 0)
        
        return markets
    
    def _calculate_late_goal_probs(self, home_times: List[int], away_times: List[int]) -> Dict[str, float]:
        """Calculate probability of goals in specific time periods"""
        total_sims = max(len(home_times) + len(away_times), 1)
        
        return {
            'goal_76-90': sum(1 for t in home_times + away_times if t >= 76) / total_sims,
            'goal_86-90': sum(1 for t in home_times + away_times if t >= 86) / total_sims,
            'home_goal_76-90': sum(1 for t in home_times if t >= 76) / (total_sims / 2),
            'away_goal_76-90': sum(1 for t in away_times if t >= 76) / (total_sims / 2)
        }

def get_state_adjusted_metrics(base_metrics: Dict[str, float], 
                               expected_state_distribution: Dict[GameState, float],
                               style: str = 'balanced') -> Dict[str, float]:
    """
    Adjust base metrics for expected game state distribution
    
    Args:
        base_metrics: Dict of metric -> base value (e.g., {'shots': 12.5, 'corners': 5.2})
        expected_state_distribution: Dict of GameState -> proportion of match in that state
        style: Team style
    
    Returns:
        Adjusted metrics
    """
    adjusted = {}
    
    for metric, base_value in base_metrics.items():
        weighted_mult = 0.0
        
        for state, proportion in expected_state_distribution.items():
            mult = getattr(DEFAULT_MULTIPLIERS[state], metric, 1.0)
            weighted_mult += mult * proportion
        
        adjusted[metric] = base_value * weighted_mult
    
    return adjusted
