"""
Opponent Strength Adjuster - Use ClubElo ratings to adjust metrics
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta

class OpponentAdjuster:
    """Adjust team metrics based on opponent quality using ClubElo ratings"""
    
    def __init__(self, hub):
        self.hub = hub
        self._elo_cache = None
        self._league_avg_elo = 1500.0  # Standard Elo baseline
    
    def get_team_strength(self, team: str, date: Optional[datetime] = None) -> float:
        """
        Get team's current Elo rating
        Args:
            team: Team name
            date: Reference date (default: today)
        Returns:
            Elo rating (typical range: 1200-2000)
        """
        try:
            if self._elo_cache is None:
                self._load_elo_data()
            
            if self._elo_cache is None or self._elo_cache.empty:
                return self._league_avg_elo
            
            if date is None:
                date = datetime.now()
            
            # Find team's most recent Elo rating before the reference date
            team_elo = self._elo_cache[
                (self._elo_cache['team'].str.contains(team, case=False, na=False)) &
                (self._elo_cache['date'] <= pd.Timestamp(date))
            ].sort_values('date', ascending=False)
            
            if not team_elo.empty:
                return float(team_elo.iloc[0]['elo'])
            
            # Fallback: Try without date filter
            team_elo_any = self._elo_cache[
                self._elo_cache['team'].str.contains(team, case=False, na=False)
            ]
            
            if not team_elo_any.empty:
                return float(team_elo_any['elo'].mean())
            
            return self._league_avg_elo
        
        except Exception as e:
            print(f"Warning: Could not get Elo rating for {team}: {e}")
            return self._league_avg_elo
    
    def adjust_metric_by_opponent(self, raw_value: float, opponent_team: str, 
                                   metric_type: str = 'offensive') -> float:
        """
        Adjust a metric based on opponent strength
        
        Args:
            raw_value: Raw metric value (e.g., xG, shots)
            opponent_team: Opponent team name
            metric_type: 'offensive' or 'defensive'
        
        Returns:
            Adjusted metric value
        """
        try:
            opponent_elo = self.get_team_strength(opponent_team)
            
            if metric_type == 'offensive':
                # Playing against stronger defense -> deflate offensive stats
                adjustment = self._league_avg_elo / opponent_elo
            else:
                # Playing against stronger attack -> inflate defensive stats allowed
                adjustment = opponent_elo / self._league_avg_elo
            
            # Cap adjustments at reasonable bounds
            adjustment = max(0.7, min(1.4, adjustment))
            
            return raw_value * adjustment
        
        except Exception as e:
            print(f"Warning: Could not adjust metric: {e}")
            return raw_value
    
    def get_strength_of_schedule(self, team: str, last_n: int = 5) -> float:
        """
        Calculate average opponent strength over last N matches
        
        Returns:
            Average opponent Elo rating
        """
        try:
            if self._elo_cache is None:
                self._load_elo_data()
            
            # Get team's recent matches from schedule
            schedule = self.hub.fbref.read_schedule()
            schedule = self._normalize_schedule(schedule)
            
            # Get matches where team played
            team_matches = schedule[
                (schedule['home_team'].str.contains(team, case=False, na=False)) |
                (schedule['away_team'].str.contains(team, case=False, na=False))
            ].sort_values('date', ascending=False).head(last_n)
            
            if team_matches.empty:
                return self._league_avg_elo
            
            # Get opponent Elo for each match
            opponent_elos = []
            for _, match in team_matches.iterrows():
                if pd.notna(match.get('home_team')) and team.lower() in str(match['home_team']).lower():
                    opponent = match['away_team']
                else:
                    opponent = match['home_team']
                
                opp_elo = self.get_team_strength(opponent, match.get('date'))
                opponent_elos.append(opp_elo)
            
            if opponent_elos:
                return float(np.mean(opponent_elos))
            
            return self._league_avg_elo
        
        except Exception as e:
            print(f"Warning: Could not calculate SoS for {team}: {e}")
            return self._league_avg_elo
    
    def get_matchup_advantage(self, home_team: str, away_team: str) -> dict:
        """
        Calculate matchup advantages
        
        Returns:
            {
                'home_elo': float,
                'away_elo': float,
                'elo_diff': float,
                'home_win_prob_elo': float,  # Based on Elo alone
                'home_advantage': float  # Home field (~100 Elo points)
            }
        """
        try:
            home_elo = self.get_team_strength(home_team)
            away_elo = self.get_team_strength(away_team)
            
            # Apply home field advantage (~100 Elo points)
            home_advantage = 100.0
            adjusted_home_elo = home_elo + home_advantage
            
            elo_diff = adjusted_home_elo - away_elo
            
            # Calculate win probability from Elo difference
            # Formula: 1 / (1 + 10^(-diff/400))
            home_win_prob = 1.0 / (1.0 + 10**(-elo_diff / 400))
            
            return {
                'home_elo': home_elo,
                'away_elo': away_elo,
                'elo_diff': elo_diff,
                'home_win_prob_elo': home_win_prob,
                'home_advantage': home_advantage
            }
        
        except Exception as e:
            print(f"Warning: Could not calculate matchup advantage: {e}")
            return {
                'home_elo': self._league_avg_elo,
                'away_elo': self._league_avg_elo,
                'elo_diff': 0.0,
                'home_win_prob_elo': 0.5,
                'home_advantage': 100.0
            }
    
    # Helper methods
    
    def _load_elo_data(self):
        """Load and cache ClubElo data"""
        try:
            elo_df = self.hub.clubelo.read_by_date()
            
            # Normalize
            if 'Date' in elo_df.columns:
                elo_df = elo_df.rename(columns={'Date': 'date'})
            if 'Team' in elo_df.columns:
                elo_df = elo_df.rename(columns={'Team': 'team'})
            if 'Elo' in elo_df.columns:
                elo_df = elo_df.rename(columns={'Elo': 'elo'})
            
            # Ensure lowercase column names
            elo_df.columns = [str(c).lower() for c in elo_df.columns]
            
            # Ensure date is datetime
            if 'date' in elo_df.columns:
                elo_df['date'] = pd.to_datetime(elo_df['date'], errors='coerce')
            
            # Filter out NaN dates
            elo_df = elo_df[elo_df['date'].notna()]
            
            self._elo_cache = elo_df
            
            # Calculate league average
            if 'elo' in elo_df.columns:
                recent_elo = elo_df[elo_df['date'] >= (datetime.now() - timedelta(days=90))]
                if not recent_elo.empty:
                    self._league_avg_elo = float(recent_elo['elo'].mean())
        
        except Exception as e:
            print(f"Warning: Could not load ClubElo data: {e}")
            self._elo_cache = None
    
    def _normalize_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize schedule DataFrame"""
        # Ensure lowercase column names
        df.columns = [str(c).lower() for c in df.columns]
        
        # Standardize team column names
        if 'team' in df.columns and 'home_team' not in df.columns:
            # Infer home/away from venue column
            if 'venue' in df.columns:
                df['home_team'] = df.apply(
                    lambda r: r['team'] if str(r.get('venue', '')).lower().startswith('home') else r.get('opponent', ''),
                    axis=1
                )
                df['away_team'] = df.apply(
                    lambda r: r['opponent'] if str(r.get('venue', '')).lower().startswith('home') else r.get('team', ''),
                    axis=1
                )
        
        # Ensure date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        return df
