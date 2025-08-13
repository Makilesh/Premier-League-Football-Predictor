import csv
import datetime
from functools import reduce
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

@dataclass
class MatchResult:
    """Structured representation of a match result"""
    date: datetime.datetime
    home_team: str
    away_team: str
    result: str
    home_goals: int
    away_goals: int
    odds_home: float
    odds_draw: float
    odds_away: float
    home_shots: int
    away_shots: int
    home_shots_target: int
    away_shots_target: int

class EnhancedDataset:
    """
    Production-grade dataset processor with time-weighted statistics,
    rest days calculation, and momentum tracking.
    """
    
    def __init__(self, file_path: str, min_matches: int = 10):
        self.raw_results: List[Dict] = []
        self.processed_results: List[Dict] = []
        self.match_objects: List[MatchResult] = []
        self.min_matches = min_matches
        self.big_6_teams = {'Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 
                           'Manchester United', 'Tottenham'}
        
        logging.info(f"Loading dataset from {file_path}")
        self._load_raw_data(file_path)
        self._create_match_objects()
        self._process_enhanced_features()
        logging.info(f"Processed {len(self.processed_results)} matches")
    
    def _load_raw_data(self, file_path: str):
        """Load raw CSV data with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                for row in reader:
                    try:
                        # Handle different date formats
                        if '/' in row['Date']:
                            row['Date'] = datetime.datetime.strptime(row['Date'], '%d/%m/%y')
                        else:
                            row['Date'] = datetime.datetime.strptime(row['Date'], '%Y-%m-%d')
                        self.raw_results.append(row)
                    except (ValueError, KeyError) as e:
                        logging.warning(f"Skipping malformed row: {e}")
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    def _create_match_objects(self):
        """Convert raw results to structured MatchResult objects"""
        for row in self.raw_results:
            try:
                # Around line 85-95, update all int() conversions:
                match = MatchResult(
                    date=row['Date'],
                    home_team=row['HomeTeam'],
                    away_team=row['AwayTeam'],
                    result=row['FTR'],
                    home_goals=int(float(row.get('FTHG', 0))),
                    away_goals=int(float(row.get('FTAG', 0))),
                    odds_home=float(row.get('B365H', 0)),
                    odds_draw=float(row.get('B365D', 0)),
                    odds_away=float(row.get('B365A', 0)),
                    home_shots=int(float(row.get('HS', 0))),
                    away_shots=int(float(row.get('AS', 0))),
                    home_shots_target=int(float(row.get('HST', 0))),
                    away_shots_target=int(float(row.get('AST', 0)))
                )

                self.match_objects.append(match)
            except (ValueError, KeyError) as e:
                logging.warning(f"Skipping match due to data error: {e}")
                continue
    
    def _process_enhanced_features(self):
        """Process matches with enhanced features"""
        for i, match in enumerate(self.match_objects):
            home_stats = self._get_enhanced_statistics(
                match.home_team, match.date, is_home=True
            )
            away_stats = self._get_enhanced_statistics(
                match.away_team, match.date, is_home=False
            )
            
            if home_stats is None or away_stats is None:
                continue
            
            # Head-to-head statistics
            h2h_stats = self._get_head_to_head_stats(
                match.home_team, match.away_team, match.date
            )
            
            # Rest days calculation
            home_rest = self._calculate_rest_days(match.home_team, match.date)
            away_rest = self._calculate_rest_days(match.away_team, match.date)
            
            # League position momentum
            home_momentum = self._calculate_momentum(match.home_team, match.date)
            away_momentum = self._calculate_momentum(match.away_team, match.date)
            
            processed_match = {
                'result': match.result,
                'odds-home': match.odds_home,
                'odds-draw': match.odds_draw,
                'odds-away': match.odds_away,
                'home-rest-days': home_rest,
                'away-rest-days': away_rest,
                'home-momentum': home_momentum,
                'away-momentum': away_momentum,
                'big-6-home': 1 if match.home_team in self.big_6_teams else 0,
                'big-6-away': 1 if match.away_team in self.big_6_teams else 0,
                'big-6-match': 1 if (match.home_team in self.big_6_teams and 
                                   match.away_team in self.big_6_teams) else 0,
            }
            
            # Add team statistics with prefixes
            for prefix, stats in [('home', home_stats), ('away', away_stats)]:
                for key, value in stats.items():
                    processed_match[f'{prefix}-{key}'] = value
            
            # Add head-to-head statistics
            for key, value in h2h_stats.items():
                processed_match[f'h2h-{key}'] = value
            
            self.processed_results.append(processed_match)
    
    def _get_enhanced_statistics(self, team: str, date: datetime.datetime, 
                               is_home: bool = True, windows: List[int] = [3, 5, 10]) -> Optional[Dict]:
        """
        Calculate time-weighted statistics across multiple rolling windows
        """
        recent_matches = self._filter_team_matches(team, date)
        
        if len(recent_matches) < self.min_matches:
            return None
        
        stats = {}
        
        # Calculate statistics for each rolling window
        for window in windows:
            if len(recent_matches) >= window:
                window_matches = recent_matches[-window:]
                window_stats = self._calculate_time_weighted_stats(
                    window_matches, team, decay_factor=0.95
                )
                
                # Add window suffix to stats
                for key, value in window_stats.items():
                    stats[f'{key}-{window}'] = value
        
        # Calculate overall weighted statistics
        overall_stats = self._calculate_time_weighted_stats(
            recent_matches[-self.min_matches:], team, decay_factor=0.9
        )
        
        for key, value in overall_stats.items():
            stats[key] = value
        
        return stats
    
    def _calculate_time_weighted_stats(self, matches: List[MatchResult], 
                                     team: str, decay_factor: float = 0.9) -> Dict:
        """
        Calculate time-weighted statistics with exponential decay
        Recent matches have higher weight
        """
        if not matches:
            return {}
        
        weights = [decay_factor ** i for i in range(len(matches)-1, -1, -1)]
        total_weight = sum(weights)
        
        stats = {
            'wins': 0, 'draws': 0, 'losses': 0,
            'goals': 0, 'goals-against': 0, 'goal-diff': 0,
            'shots': 0, 'shots-target': 0, 'shots-against': 0,
            'shots-target-against': 0, 'shot-accuracy': 0,
            'points': 0, 'home-advantage': 0
        }
        
        for match, weight in zip(matches, weights):
            is_home = match.home_team == team
            
            if is_home:
                goals_for = match.home_goals
                goals_against = match.away_goals
                shots_for = match.home_shots
                shots_target_for = match.home_shots_target
                shots_against = match.away_shots
                shots_target_against = match.away_shots_target
            else:
                goals_for = match.away_goals
                goals_against = match.home_goals
                shots_for = match.away_shots
                shots_target_for = match.away_shots_target
                shots_against = match.home_shots
                shots_target_against = match.home_shots_target
            
            # Match result from team perspective
            if (is_home and match.result == 'H') or (not is_home and match.result == 'A'):
                stats['wins'] += weight
                stats['points'] += 3 * weight
            elif match.result == 'D':
                stats['draws'] += weight
                stats['points'] += 1 * weight
            else:
                stats['losses'] += weight
            
            stats['goals'] += goals_for * weight
            stats['goals-against'] += goals_against * weight
            stats['goal-diff'] += (goals_for - goals_against) * weight
            stats['shots'] += shots_for * weight
            stats['shots-target'] += shots_target_for * weight
            stats['shots-against'] += shots_against * weight
            stats['shots-target-against'] += shots_target_against * weight
            
            if is_home:
                stats['home-advantage'] += weight
        
        # Normalize by total weight
        for key in stats:
            stats[key] /= total_weight
        
        # Calculate derived metrics
        if stats['shots'] > 0:
            stats['shot-accuracy'] = stats['shots-target'] / stats['shots']
        
        return stats
    
    def _filter_team_matches(self, team: str, date: datetime.datetime) -> List[MatchResult]:
        """Filter matches for a team before a given date"""
        # Ensure date is timezone-naive for comparison
        if hasattr(date, 'tzinfo') and date.tzinfo is not None:
            date = date.replace(tzinfo=None)
        
        return [match for match in self.match_objects 
                if (match.home_team == team or match.away_team == team) 
                and match.date < date]
    
    def _calculate_rest_days(self, team: str, match_date: datetime.datetime) -> int:
        """Calculate days since last match"""
        # Ensure match_date is timezone-naive for comparison
        if hasattr(match_date, 'tzinfo') and match_date.tzinfo is not None:
            match_date = match_date.replace(tzinfo=None)
            
        team_matches = self._filter_team_matches(team, match_date)
        
        if not team_matches:
            return 7  # Default rest days
        
        last_match_date = team_matches[-1].date
        rest_days = (match_date - last_match_date).days
        
        return min(rest_days, 14)  # Cap at 14 days for consistency
    
    def _calculate_momentum(self, team: str, match_date: datetime.datetime, 
                          window: int = 5) -> float:
        """
        Calculate team momentum based on recent point accumulation rate
        Positive values indicate upward trend, negative indicates decline
        """
        team_matches = self._filter_team_matches(team, match_date)
        
        if len(team_matches) < window * 2:
            return 0.0
        
        # Compare recent performance to earlier performance
        recent_matches = team_matches[-window:]
        earlier_matches = team_matches[-window*2:-window]
        
        recent_points = sum(self._get_match_points(match, team) for match in recent_matches)
        earlier_points = sum(self._get_match_points(match, team) for match in earlier_matches)
        
        momentum = (recent_points - earlier_points) / window
        
        return max(-3, min(3, momentum))  # Normalize between -3 and 3
    
    def _get_match_points(self, match: MatchResult, team: str) -> int:
        """Get points earned by team in a match"""
        is_home = match.home_team == team
        
        if (is_home and match.result == 'H') or (not is_home and match.result == 'A'):
            return 3
        elif match.result == 'D':
            return 1
        else:
            return 0
    
    def _get_head_to_head_stats(self, home_team: str, away_team: str, 
                               match_date: datetime.datetime, 
                               lookback_years: int = 3) -> Dict:
        """
        Calculate head-to-head statistics between two teams
        """
        # Ensure match_date is timezone-naive for comparison
        if hasattr(match_date, 'tzinfo') and match_date.tzinfo is not None:
            match_date = match_date.replace(tzinfo=None)
            
        cutoff_date = match_date - datetime.timedelta(days=365 * lookback_years)
        
        h2h_matches = [
            match for match in self.match_objects
            if match.date >= cutoff_date and match.date < match_date
            and ((match.home_team == home_team and match.away_team == away_team) or
                 (match.home_team == away_team and match.away_team == home_team))
        ]
        
        if not h2h_matches:
            return {
                'matches': 0, 'home-wins': 0, 'away-wins': 0, 'draws': 0,
                'home-goals-avg': 0, 'away-goals-avg': 0
            }
        
        stats = {
            'matches': len(h2h_matches),
            'home-wins': 0, 'away-wins': 0, 'draws': 0,
            'home-goals-total': 0, 'away-goals-total': 0
        }
        
        for match in h2h_matches:
            if match.home_team == home_team:
                # Home team is our focus home team
                if match.result == 'H':
                    stats['home-wins'] += 1
                elif match.result == 'A':
                    stats['away-wins'] += 1
                else:
                    stats['draws'] += 1
                stats['home-goals-total'] += match.home_goals
                stats['away-goals-total'] += match.away_goals
            else:
                # Teams are reversed
                if match.result == 'A':
                    stats['home-wins'] += 1
                elif match.result == 'H':
                    stats['away-wins'] += 1
                else:
                    stats['draws'] += 1
                stats['home-goals-total'] += match.away_goals
                stats['away-goals-total'] += match.home_goals
        
        # Calculate averages
        stats['home-goals-avg'] = stats['home-goals-total'] / len(h2h_matches)
        stats['away-goals-avg'] = stats['away-goals-total'] / len(h2h_matches)
        
        # Remove totals from final stats
        del stats['home-goals-total']
        del stats['away-goals-total']
        
        return stats
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names for model training"""
        if not self.processed_results:
            return []
        
        # Get all keys except the target variable
        feature_names = [key for key in self.processed_results[0].keys() 
                        if key not in ['result']]
        
        return sorted(feature_names)
    
    def get_training_data(self, train_fraction: float = 0.8) -> Tuple[Dict, np.ndarray, Dict, np.ndarray]:
        """
        Split data into training and testing sets
        Returns: (train_features, train_labels, test_features, test_labels)
        """
        if not self.processed_results:
            raise ValueError("No processed results available")
        
        split_idx = int(len(self.processed_results) * train_fraction)
        train_data = self.processed_results[:split_idx]
        test_data = self.processed_results[split_idx:]
        
        def extract_features_labels(data):
            features = {}
            labels = []
            
            for match in data:
                labels.append(match['result'])
                for key, value in match.items():
                    if key != 'result':
                        if key not in features:
                            features[key] = []
                        features[key].append(value)
            
            # Convert to numpy arrays
            for key in features:
                features[key] = np.array(features[key], dtype=np.float32)
            
            return features, np.array(labels)
        
        train_features, train_labels = extract_features_labels(train_data)
        test_features, test_labels = extract_features_labels(test_data)
        
        return train_features, train_labels, test_features, test_labels
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the dataset"""
        if not self.processed_results:
            return {}
        
        total_matches = len(self.processed_results)
        results = [match['result'] for match in self.processed_results]
        
        return {
            'total_matches': total_matches,
            'home_wins': results.count('H'),
            'draws': results.count('D'),
            'away_wins': results.count('A'),
            'home_win_pct': results.count('H') / total_matches * 100,
            'draw_pct': results.count('D') / total_matches * 100,
            'away_win_pct': results.count('A') / total_matches * 100,
            'feature_count': len(self.get_feature_names()),
            'date_range': (
                min(match.date for match in self.match_objects),
                max(match.date for match in self.match_objects)
            )
        }

# Usage example and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize enhanced dataset
    dataset = EnhancedDataset('data/combined_all_seasons.csv')
    
    # Print summary
    summary = dataset.get_data_summary()
    print("\nDataset Summary:")
    print(f"Total matches: {summary['total_matches']}")
    print(f"Home wins: {summary['home_wins']} ({summary['home_win_pct']:.1f}%)")
    print(f"Draws: {summary['draws']} ({summary['draw_pct']:.1f}%)")
    print(f"Away wins: {summary['away_wins']} ({summary['away_win_pct']:.1f}%)")
    print(f"Features: {summary['feature_count']}")
    print(f"Date range: {summary['date_range'][0].strftime('%Y-%m-%d')} to {summary['date_range'][1].strftime('%Y-%m-%d')}")
    
    # Show feature names
    print("\nNew Features Added:")
    features = dataset.get_feature_names()
    enhanced_features = [f for f in features if any(keyword in f for keyword in 
                        ['rest', 'momentum', 'big-6', 'h2h', '-3', '-5', '-10'])]
    for feature in enhanced_features:
        print(f"- {feature}")
    
    print(f"\nTotal features: {len(features)}")