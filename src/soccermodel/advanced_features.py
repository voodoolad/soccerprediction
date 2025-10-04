"""
Advanced Feature Engine - Extract metrics beyond basic stats
Includes: NP-xG, PPDA, set pieces, transition metrics
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional

class AdvancedFeatureEngine:
    """Extract advanced metrics from multiple data sources"""
    
    def __init__(self, hub):
        self.hub = hub
        self._understat_cache = None
        self._defense_cache = None
        self._shooting_cache = None
    
    def get_npxg_adjusted(self, team: str, opponent: str, last_n: int = 5) -> Dict[str, float]:
        """
        Get opponent-adjusted non-penalty xG
        Returns: {'npxg': float, 'npxga': float, 'npxg_per90': float}
        """
        try:
            if self._understat_cache is None:
                understat_raw = self.hub.understat.read_team_match_stats()
                self._understat_cache = self._normalize_df(understat_raw)
            
            df = self._understat_cache
            team_matches = df[df['team'].str.contains(team, case=False, na=False)].sort_values('date').tail(last_n)
            
            if team_matches.empty:
                return {'npxg': np.nan, 'npxga': np.nan, 'npxg_per90': np.nan}
            
            # Find xG columns (Understat uses 'xG', 'npxG', 'xGA', 'npxGA')
            npxg_col = self._find_col(team_matches, ['npxg', 'np:xg', 'non-penalty xg'])
            npxga_col = self._find_col(team_matches, ['npxga', 'np:xga', 'non-penalty xga'])
            
            npxg_values = pd.to_numeric(team_matches[npxg_col], errors='coerce') if npxg_col else None
            npxga_values = pd.to_numeric(team_matches[npxga_col], errors='coerce') if npxga_col else None
            
            # Apply recency weights
            weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])[:len(team_matches)]
            weights = weights / weights.sum()
            
            result = {}
            if npxg_values is not None and npxg_values.notna().any():
                result['npxg'] = float(np.average(npxg_values.fillna(0), weights=weights))
            else:
                result['npxg'] = np.nan
            
            if npxga_values is not None and npxga_values.notna().any():
                result['npxga'] = float(np.average(npxga_values.fillna(0), weights=weights))
            else:
                result['npxga'] = np.nan
            
            # Opponent adjustment using simple league average
            if not np.isnan(result['npxg']):
                league_avg_defense = self._get_league_avg_defense()
                opponent_defense_rating = self._get_team_defense_rating(opponent)
                
                if opponent_defense_rating > 0:
                    result['npxg_adjusted'] = result['npxg'] * (league_avg_defense / opponent_defense_rating)
                else:
                    result['npxg_adjusted'] = result['npxg']
            else:
                result['npxg_adjusted'] = np.nan
            
            result['npxg_per90'] = result['npxg'] if not np.isnan(result['npxg']) else np.nan
            
            return result
        
        except Exception as e:
            print(f"Warning: Could not extract NP-xG for {team}: {e}")
            return {'npxg': np.nan, 'npxga': np.nan, 'npxg_per90': np.nan}
    
    def get_ppda(self, team: str, last_n: int = 5) -> float:
        """
        Get PPDA (Passes Per Defensive Action) - lower = more aggressive press
        Returns: float (typical range 7-15)
        """
        try:
            if self._defense_cache is None:
                defense_raw = self.hub.fbref.read_team_match_stats(stat_type="defense", opponent_stats=False)
                self._defense_cache = self._normalize_df(defense_raw)
            
            df = self._defense_cache
            team_matches = df[df['team'].str.contains(team, case=False, na=False)].sort_values('date').tail(last_n)
            
            if team_matches.empty:
                return np.nan
            
            # PPDA column in FBref defense stats
            ppda_col = self._find_col(team_matches, ['ppda', 'passes per defensive action', 'press'])
            
            if ppda_col:
                ppda_values = pd.to_numeric(team_matches[ppda_col], errors='coerce')
                if ppda_values.notna().any():
                    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])[:len(team_matches)]
                    weights = weights / weights.sum()
                    return float(np.average(ppda_values.fillna(ppda_values.mean()), weights=weights))
            
            # Fallback: calculate from pressures and opponent passes if available
            pressures_col = self._find_col(team_matches, ['press', 'pressures'])
            if pressures_col:
                # Simplified PPDA proxy
                return 12.0  # League average fallback
            
            return np.nan
        
        except Exception as e:
            print(f"Warning: Could not extract PPDA for {team}: {e}")
            return np.nan
    
    def get_set_piece_threat(self, team: str, last_n: int = 5) -> Dict[str, float]:
        """
        Get set piece xG for/against
        Returns: {'sp_xg_for': float, 'sp_xg_against': float}
        """
        try:
            if self._shooting_cache is None:
                shooting_raw = self.hub.fbref.read_team_match_stats(stat_type="shooting", opponent_stats=False)
                self._shooting_cache = self._normalize_df(shooting_raw)
            
            df = self._shooting_cache
            team_matches = df[df['team'].str.contains(team, case=False, na=False)].sort_values('date').tail(last_n)
            
            if team_matches.empty:
                return {'sp_xg_for': np.nan, 'sp_xg_against': np.nan}
            
            # Look for set piece xG columns (if available in FBref)
            # Note: This may not be directly available; may need to infer from other stats
            
            return {
                'sp_xg_for': np.nan,  # Placeholder - need to identify correct column
                'sp_xg_against': np.nan
            }
        
        except Exception as e:
            print(f"Warning: Could not extract set piece data for {team}: {e}")
            return {'sp_xg_for': np.nan, 'sp_xg_against': np.nan}
    
    def get_transition_metrics(self, team: str, last_n: int = 5) -> Dict[str, float]:
        """
        Get transition (counter-attack) metrics
        Returns: {'prog_passes': float, 'prog_carries': float, 'counter_xg': float}
        """
        try:
            # Use FBref possession and passing stats
            poss_raw = self.hub.fbref.read_team_match_stats(stat_type="possession", opponent_stats=False)
            poss_df = self._normalize_df(poss_raw)
            
            team_matches = poss_df[poss_df['team'].str.contains(team, case=False, na=False)].sort_values('date').tail(last_n)
            
            if team_matches.empty:
                return {'prog_passes': np.nan, 'prog_carries': np.nan}
            
            # Progressive passes and carries
            prog_pass_col = self._find_col(team_matches, ['progressive passes', 'prog', 'prg'])
            prog_carry_col = self._find_col(team_matches, ['progressive carries', 'carries', 'prg c'])
            
            weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])[:len(team_matches)]
            weights = weights / weights.sum()
            
            result = {}
            
            if prog_pass_col:
                vals = pd.to_numeric(team_matches[prog_pass_col], errors='coerce')
                result['prog_passes'] = float(np.average(vals.fillna(0), weights=weights))
            else:
                result['prog_passes'] = np.nan
            
            if prog_carry_col:
                vals = pd.to_numeric(team_matches[prog_carry_col], errors='coerce')
                result['prog_carries'] = float(np.average(vals.fillna(0), weights=weights))
            else:
                result['prog_carries'] = np.nan
            
            return result
        
        except Exception as e:
            print(f"Warning: Could not extract transition metrics for {team}: {e}")
            return {'prog_passes': np.nan, 'prog_carries': np.nan}
    
    # Helper methods
    
    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns and ensure required fields"""
        if not isinstance(df.columns, pd.MultiIndex):
            # Flatten if needed
            pass
        
        # Ensure lowercase column names for easier matching
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # Ensure date column
        if 'date' not in df.columns:
            for col in df.columns:
                if 'date' in col or 'time' in col:
                    try:
                        df['date'] = pd.to_datetime(df[col], errors='coerce')
                        break
                    except:
                        pass
        
        return df
    
    def _find_col(self, df: pd.DataFrame, aliases: list) -> Optional[str]:
        """Find column matching any alias (case-insensitive)"""
        cols_lower = {c.lower(): c for c in df.columns}
        for alias in aliases:
            for col_lower, col_actual in cols_lower.items():
                if alias.lower() in col_lower:
                    return col_actual
        return None
    
    def _get_league_avg_defense(self) -> float:
        """Get league average defensive rating (npxGA)"""
        try:
            if self._understat_cache is None:
                return 1.3  # Fallback
            
            npxga_col = self._find_col(self._understat_cache, ['npxga', 'np:xga'])
            if npxga_col:
                return float(pd.to_numeric(self._understat_cache[npxga_col], errors='coerce').mean(skipna=True))
            return 1.3
        except:
            return 1.3
    
    def _get_team_defense_rating(self, team: str) -> float:
        """Get team's defensive rating (npxGA conceded)"""
        try:
            if self._understat_cache is None:
                return 1.3
            
            df = self._understat_cache
            team_matches = df[df['team'].str.contains(team, case=False, na=False)].tail(10)
            
            npxga_col = self._find_col(team_matches, ['npxga', 'np:xga'])
            if npxga_col:
                vals = pd.to_numeric(team_matches[npxga_col], errors='coerce')
                if vals.notna().any():
                    return float(vals.mean(skipna=True))
            
            return 1.3
        except:
            return 1.3
