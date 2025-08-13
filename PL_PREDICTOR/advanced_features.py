import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import datetime
from dataclasses import dataclass
from enhanced_dataset import EnhancedDataset, MatchResult
import logging

class AdvancedFeatureEngine:
    """
    Advanced feature engineering for profitable football betting.
    Focuses on features that actually predict outcomes, not just correlate.
    """
    
    def __init__(self, dataset: EnhancedDataset):
        self.dataset = dataset
        self.match_objects = dataset.match_objects
        self.season_boundaries = self._identify_season_boundaries()
        
    def _identify_season_boundaries(self) -> Dict[int, Tuple[datetime.datetime, datetime.datetime]]:
        """Identify season start/end dates for motivation calculations"""
        seasons = {}
        
        for match in self.match_objects:
            # Premier League season runs Aug-May
            if match.date.month >= 8:
                season_year = match.date.year
            else:
                season_year = match.date.year - 1
                
            if season_year not in seasons:
                seasons[season_year] = [match.date, match.date]
            else:
                seasons[season_year][0] = min(seasons[season_year][0], match.date)
                seasons[season_year][1] = max(seasons[season_year][1], match.date)
        
        return {year: (start, end) for year, (start, end) in seasons.items()}
    
    def generate_advanced_features(self) -> List[Dict]:
        """Generate all advanced features for each match"""
        advanced_features = []
        
        logging.info("Generating advanced features...")
        
        for i, match in enumerate(self.match_objects):
            if i % 500 == 0:
                logging.info(f"Processing match {i}/{len(self.match_objects)}")
                
            features = self._extract_match_features(match)
            
            if features is not None:
                advanced_features.append(features)
        
        logging.info(f"Generated {len(advanced_features)} feature sets")
        return advanced_features
    
    def _extract_match_features(self, match: MatchResult) -> Optional[Dict]:
        """Extract all advanced features for a single match"""
        
        # Get basic team stats (minimum matches required)
        home_matches = self._get_team_matches(match.home_team, match.date)
        away_matches = self._get_team_matches(match.away_team, match.date)
        
        if len(home_matches) < 10 or len(away_matches) < 10:
            return None
        
        features = {
            'result': match.result,
            'odds-home': match.odds_home,
            'odds-draw': match.odds_draw,
            'odds-away': match.odds_away,
        }
        
        # 1. FORM ANALYSIS - Multiple Windows
        home_form = self._calculate_form_features(home_matches, match.home_team)
        away_form = self._calculate_form_features(away_matches, match.away_team)
        
        for key, value in home_form.items():
            features[f'home-{key}'] = value
        for key, value in away_form.items():
            features[f'away-{key}'] = value
        
        # 2. GOAL DIFFERENCE TRENDS
        home_gd_trend = self._calculate_goal_difference_trend(home_matches, match.home_team)
        away_gd_trend = self._calculate_goal_difference_trend(away_matches, match.away_team)
        
        features['home-gd-trend'] = home_gd_trend
        features['away-gd-trend'] = away_gd_trend
        features['gd-trend-diff'] = home_gd_trend - away_gd_trend
        
        # 3. BIG 6 DYNAMICS
        big_6_features = self._calculate_big6_features(match, home_matches, away_matches)
        features.update(big_6_features)
        
        # 4. SEASONAL MOTIVATION
        motivation_features = self._calculate_motivation_features(
            match, home_matches, away_matches
        )
        features.update(motivation_features)
        
        # 5. TACTICAL PATTERNS
        tactical_features = self._calculate_tactical_features(
            match, home_matches, away_matches
        )
        features.update(tactical_features)
        
        # 6. PRESSURE SITUATIONS
        pressure_features = self._calculate_pressure_features(
            match, home_matches, away_matches
        )
        features.update(pressure_features)
        
        # 7. MARKET INEFFICIENCY INDICATORS
        market_features = self._calculate_market_features(match)
        features.update(market_features)
        
        return features
    
    def _get_team_matches(self, team: str, before_date: datetime.datetime) -> List[MatchResult]:
        """Get all matches for a team before a given date"""
        return [match for match in self.match_objects
                if (match.home_team == team or match.away_team == team)
                and match.date < before_date]
    
    def _calculate_form_features(self, matches: List[MatchResult], team: str) -> Dict:
        """Calculate comprehensive form features across multiple windows"""
        features = {}
        
        windows = [3, 5, 10]
        
        for window in windows:
            if len(matches) >= window:
                recent_matches = matches[-window:]
                
                # Basic form metrics
                points = 0
                goals_for = 0
                goals_against = 0
                shots_for = 0
                shots_against = 0
                home_games = 0
                
                for match in recent_matches:
                    is_home = match.home_team == team
                    
                    if is_home:
                        team_goals = match.home_goals
                        opp_goals = match.away_goals
                        team_shots = match.home_shots
                        opp_shots = match.away_shots
                        home_games += 1
                    else:
                        team_goals = match.away_goals
                        opp_goals = match.home_goals
                        team_shots = match.away_shots
                        opp_shots = match.home_shots
                    
                    goals_for += team_goals
                    goals_against += opp_goals
                    shots_for += team_shots
                    shots_against += opp_shots
                    
                    # Points calculation
                    if (is_home and match.result == 'H') or (not is_home and match.result == 'A'):
                        points += 3
                    elif match.result == 'D':
                        points += 1
                
                # Calculate derived metrics
                features[f'ppg-{window}'] = points / window
                features[f'gpg-{window}'] = goals_for / window
                features[f'gapg-{window}'] = goals_against / window
                features[f'gd-{window}'] = (goals_for - goals_against) / window
                features[f'spg-{window}'] = shots_for / window
                features[f'sapg-{window}'] = shots_against / window
                features[f'home-ratio-{window}'] = home_games / window
                
                # Shot conversion rate
                if shots_for > 0:
                    features[f'conversion-{window}'] = goals_for / shots_for
                else:
                    features[f'conversion-{window}'] = 0
        
        return features
    
    def _calculate_goal_difference_trend(self, matches: List[MatchResult], team: str, 
                                       window: int = 8) -> float:
        """Calculate goal difference trend over recent matches"""
        if len(matches) < window:
            return 0.0
        
        recent_matches = matches[-window:]
        goal_diffs = []
        
        for match in recent_matches:
            is_home = match.home_team == team
            
            if is_home:
                gd = match.home_goals - match.away_goals
            else:
                gd = match.away_goals - match.home_goals
            
            goal_diffs.append(gd)
        
        # Calculate trend using linear regression slope
        x = np.arange(len(goal_diffs))
        y = np.array(goal_diffs)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        
        return 0.0
    
    def _calculate_big6_features(self, match: MatchResult, 
                               home_matches: List[MatchResult], 
                               away_matches: List[MatchResult]) -> Dict:
        """Calculate Big 6 specific features"""
        big_6 = {'Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 
                'Manchester United', 'Tottenham'}
        
        features = {
            'big6-home': 1 if match.home_team in big_6 else 0,
            'big6-away': 1 if match.away_team in big_6 else 0,
            'big6-clash': 1 if (match.home_team in big_6 and match.away_team in big_6) else 0,
        }
        
        # Performance vs Big 6 teams
        def vs_big6_record(matches, team):
            vs_big6_matches = [m for m in matches[-20:] if 
                             (m.home_team in big_6 and m.home_team != team) or
                             (m.away_team in big_6 and m.away_team != team)]
            
            if not vs_big6_matches:
                return {'points': 0, 'goals': 0, 'conceded': 0}
            
            points, goals, conceded = 0, 0, 0
            
            for m in vs_big6_matches:
                is_home = m.home_team == team
                
                if is_home:
                    team_goals, opp_goals = m.home_goals, m.away_goals
                else:
                    team_goals, opp_goals = m.away_goals, m.home_goals
                
                goals += team_goals
                conceded += opp_goals
                
                if (is_home and m.result == 'H') or (not is_home and m.result == 'A'):
                    points += 3
                elif m.result == 'D':
                    points += 1
            
            return {
                'points': points / len(vs_big6_matches),
                'goals': goals / len(vs_big6_matches), 
                'conceded': conceded / len(vs_big6_matches)
            }
        
        home_vs_big6 = vs_big6_record(home_matches, match.home_team)
        away_vs_big6 = vs_big6_record(away_matches, match.away_team)
        
        features['home-vs-big6-ppg'] = home_vs_big6['points']
        features['away-vs-big6-ppg'] = away_vs_big6['points']
        features['home-vs-big6-gpg'] = home_vs_big6['goals']
        features['away-vs-big6-gpg'] = away_vs_big6['goals']
        
        return features
    
    def _calculate_motivation_features(self, match: MatchResult,
                                     home_matches: List[MatchResult],
                                     away_matches: List[MatchResult]) -> Dict:
        """Calculate end-season motivation factors"""
        features = {}
        
        # Determine season and position in season
        if match.date.month >= 8:
            season_year = match.date.year
        else:
            season_year = match.date.year - 1
        
        if season_year not in self.season_boundaries:
            return {'season-progress': 0.5, 'end-season-pressure': 0}
        
        season_start, season_end = self.season_boundaries[season_year]
        total_season_days = (season_end - season_start).days
        elapsed_days = (match.date - season_start).days
        
        season_progress = elapsed_days / total_season_days
        features['season-progress'] = min(1.0, max(0.0, season_progress))
        
        # End-of-season pressure (higher in last 10 games)
        if season_progress > 0.8:
            features['end-season-pressure'] = (season_progress - 0.8) * 5
        else:
            features['end-season-pressure'] = 0
        
        # Calculate league positions (approximate from recent form)
        def estimate_position(matches, team):
            if len(matches) < 20:
                return 10  # Mid-table default
            
            recent_ppg = 0
            for m in matches[-20:]:
                is_home = m.home_team == team
                if (is_home and m.result == 'H') or (not is_home and m.result == 'A'):
                    recent_ppg += 3
                elif m.result == 'D':
                    recent_ppg += 1
            
            recent_ppg /= 20
            
            # Convert PPG to approximate league position (very rough)
            if recent_ppg > 2.0:
                return np.random.randint(1, 6)    # Top 6
            elif recent_ppg > 1.5:
                return np.random.randint(7, 12)   # Mid-table
            else:
                return np.random.randint(13, 20)  # Bottom half
        
        home_pos = estimate_position(home_matches, match.home_team)
        away_pos = estimate_position(away_matches, match.away_team)
        
        features['home-position'] = home_pos
        features['away-position'] = away_pos
        features['position-diff'] = away_pos - home_pos  # Positive means home team higher
        
        # Relegation battle (bottom 6 desperation)
        features['home-relegation-fight'] = 1 if home_pos >= 15 else 0
        features['away-relegation-fight'] = 1 if away_pos >= 15 else 0
        
        # European competition chase (top 7)
        features['home-europe-chase'] = 1 if home_pos <= 7 else 0
        features['away-europe-chase'] = 1 if away_pos <= 7 else 0
        
        return features
    
    def _calculate_tactical_features(self, match: MatchResult,
                                   home_matches: List[MatchResult],
                                   away_matches: List[MatchResult]) -> Dict:
        """Calculate tactical pattern features"""
        features = {}
        
        def calculate_style_metrics(matches, team):
            if len(matches) < 8:
                return {}
            
            recent_matches = matches[-8:]
            
            total_shots = 0
            total_goals = 0
            total_shots_against = 0
            total_goals_against = 0
            possession_proxy = 0  # shots ratio as possession proxy
            
            for m in recent_matches:
                is_home = m.home_team == team
                
                if is_home:
                    shots_for = m.home_shots
                    goals_for = m.home_goals
                    shots_against = m.away_shots
                    goals_against = m.away_goals
                else:
                    shots_for = m.away_shots
                    goals_for = m.away_goals
                    shots_against = m.home_shots
                    goals_against = m.home_goals
                
                total_shots += shots_for
                total_goals += goals_for
                total_shots_against += shots_against
                total_goals_against += goals_against
                
                if shots_for + shots_against > 0:
                    possession_proxy += shots_for / (shots_for + shots_against)
            
            return {
                'shots_per_game': total_shots / len(recent_matches),
                'goals_per_game': total_goals / len(recent_matches),
                'shots_against_per_game': total_shots_against / len(recent_matches),
                'conversion_rate': total_goals / total_shots if total_shots > 0 else 0,
                'possession_proxy': possession_proxy / len(recent_matches)
            }
        
        home_style = calculate_style_metrics(home_matches, match.home_team)
        away_style = calculate_style_metrics(away_matches, match.away_team)
        
        for key, value in home_style.items():
            features[f'home-{key}'] = value
        for key, value in away_style.items():
            features[f'away-{key}'] = value
        
        # Style clash indicators
        if home_style and away_style:
            features['possession-diff'] = home_style['possession_proxy'] - away_style['possession_proxy']
            features['shots-ratio'] = (home_style['shots_per_game'] / 
                                     max(0.1, away_style['shots_against_per_game']))
            features['defensive-clash'] = min(home_style['shots_against_per_game'],
                                            away_style['shots_against_per_game'])
        
        return features
    
    def _calculate_pressure_features(self, match: MatchResult,
                                   home_matches: List[MatchResult],
                                   away_matches: List[MatchResult]) -> Dict:
        """Calculate pressure situation features"""
        features = {}
        
        def recent_form_pressure(matches, team):
            if len(matches) < 5:
                return 0
            
            recent_results = matches[-5:]
            points = 0
            
            for m in recent_results:
                is_home = m.home_team == team
                if (is_home and m.result == 'H') or (not is_home and m.result == 'A'):
                    points += 3
                elif m.result == 'D':
                    points += 1
            
            expected_points = 2.0 * len(recent_results)  # 2 PPG is decent
            pressure = max(0, expected_points - points) / expected_points
            return pressure
        
        features['home-form-pressure'] = recent_form_pressure(home_matches, match.home_team)
        features['away-form-pressure'] = recent_form_pressure(away_matches, match.away_team)
        
        # Streak analysis
        def current_streak(matches, team):
            if not matches:
                return {'type': 'none', 'length': 0}
            
            streak_type = None
            streak_length = 0
            
            for m in reversed(matches[-10:]):  # Last 10 matches
                is_home = m.home_team == team
                
                if (is_home and m.result == 'H') or (not is_home and m.result == 'A'):
                    current_result = 'win'
                elif m.result == 'D':
                    current_result = 'draw'
                else:
                    current_result = 'loss'
                
                if streak_type is None:
                    streak_type = current_result
                    streak_length = 1
                elif streak_type == current_result:
                    streak_length += 1
                else:
                    break
            
            return {'type': streak_type, 'length': streak_length}
        
        home_streak = current_streak(home_matches, match.home_team)
        away_streak = current_streak(away_matches, match.away_team)
        
        features['home-win-streak'] = home_streak['length'] if home_streak['type'] == 'win' else 0
        features['home-loss-streak'] = home_streak['length'] if home_streak['type'] == 'loss' else 0
        features['away-win-streak'] = away_streak['length'] if away_streak['type'] == 'win' else 0
        features['away-loss-streak'] = away_streak['length'] if away_streak['type'] == 'loss' else 0
        
        return features
    
    def _calculate_market_features(self, match: MatchResult) -> Dict:
        """Calculate market inefficiency indicators"""
        features = {}
        
        # Odds analysis
        if match.odds_home > 0 and match.odds_draw > 0 and match.odds_away > 0:
            # Implied probabilities
            prob_home = 1 / match.odds_home
            prob_draw = 1 / match.odds_draw
            prob_away = 1 / match.odds_away
            
            total_prob = prob_home + prob_draw + prob_away
            overround = total_prob - 1  # Bookmaker margin
            
            # Normalized probabilities
            norm_prob_home = prob_home / total_prob
            norm_prob_draw = prob_draw / total_prob
            norm_prob_away = prob_away / total_prob
            
            features['market-overround'] = overround
            features['market-prob-home'] = norm_prob_home
            features['market-prob-draw'] = norm_prob_draw
            features['market-prob-away'] = norm_prob_away
            
            # Market confidence indicators
            features['market-favorite-odds'] = min(match.odds_home, match.odds_draw, match.odds_away)
            features['market-underdog-odds'] = max(match.odds_home, match.odds_draw, match.odds_away)
            features['market-spread'] = features['market-underdog-odds'] - features['market-favorite-odds']
            
            # Draw value indicator
            features['draw-value'] = match.odds_draw / (match.odds_home + match.odds_away) * 2
            
        return features

class AdvancedDatasetProcessor:
    """
    Combines enhanced dataset with advanced feature engineering
    """
    
    def __init__(self, file_path: str):
        self.base_dataset = EnhancedDataset(file_path)
        self.feature_engine = AdvancedFeatureEngine(self.base_dataset)
        self.processed_data = None
        
    def process_all_features(self) -> List[Dict]:
        """Process all features - base + advanced"""
        if self.processed_data is None:
            # Get base processed results
            base_features = self.base_dataset.processed_results
            
            # Get advanced features
            advanced_features = self.feature_engine.generate_advanced_features()
            
            # Merge features (match by index)
            merged_data = []
            min_length = min(len(base_features), len(advanced_features))
            
            for i in range(min_length):
                merged_match = {**base_features[i]}  # Start with base features
                
                # Add advanced features (skip duplicates)
                for key, value in advanced_features[i].items():
                    if key not in merged_match:
                        merged_match[key] = value
                
                merged_data.append(merged_match)
            
            self.processed_data = merged_data
            
        return self.processed_data
    
    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """Categorize features by type for analysis"""
        if not self.processed_data:
            return {}
        
        sample = self.processed_data[0]
        categories = {
            'form_features': [],
            'tactical_features': [],
            'motivation_features': [],
            'big6_features': [],
            'pressure_features': [],
            'market_features': [],
            'basic_features': []
        }
        
        for feature in sample.keys():
            if feature in ['result', 'odds-home', 'odds-draw', 'odds-away']:
                continue
                
            if any(x in feature for x in ['ppg', 'gpg', 'gd-', 'conversion']):
                categories['form_features'].append(feature)
            elif any(x in feature for x in ['shots', 'possession', 'style']):
                categories['tactical_features'].append(feature)
            elif any(x in feature for x in ['season', 'position', 'europe', 'relegation']):
                categories['motivation_features'].append(feature)
            elif 'big' in feature or 'vs-big6' in feature:
                categories['big6_features'].append(feature)
            elif any(x in feature for x in ['pressure', 'streak', 'momentum']):
                categories['pressure_features'].append(feature)
            elif 'market' in feature:
                categories['market_features'].append(feature)
            else:
                categories['basic_features'].append(feature)
        
        return categories
    
    def get_training_data(self, train_fraction: float = 0.8) -> Tuple[Dict, np.ndarray, Dict, np.ndarray]:
        """Get training data with all features"""
        data = self.process_all_features()
        
        split_idx = int(len(data) * train_fraction)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        def extract_features_labels(data_subset):
            features = {}
            labels = []
            
            for match in data_subset:
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
    
    def get_summary(self) -> Dict:
        """Get comprehensive dataset summary"""
        data = self.process_all_features()
        categories = self.get_feature_importance_categories()
        
        total_features = sum(len(features) for features in categories.values())
        
        return {
            'total_matches': len(data),
            'total_features': total_features,
            'feature_breakdown': {cat: len(features) for cat, features in categories.items()},
            'sample_advanced_features': {
                'Form Features': categories['form_features'][:5],
                'Tactical Features': categories['tactical_features'][:5], 
                'Market Features': categories['market_features'][:3],
                'Pressure Features': categories['pressure_features'][:3]
            }
        }

# Usage example and integration
def create_production_dataset(file_path: str = 'data/combined_all_seasons.csv'):
    """Create production-ready dataset with all advanced features"""
    processor = AdvancedDatasetProcessor(file_path)
    
    print("🚀 Processing Advanced Features...")
    summary = processor.get_summary()
    
    print(f"📊 Advanced Dataset Summary:")
    print(f"Total matches: {summary['total_matches']}")
    print(f"Total features: {summary['total_features']}")
    print(f"\nFeature Breakdown:")
    for category, count in summary['feature_breakdown'].items():
        print(f"  {category.replace('_', ' ').title()}: {count}")
    
    print(f"\n🎯 Sample Advanced Features:")
    for category, features in summary['sample_advanced_features'].items():
        print(f"{category}: {', '.join(features)}")
    
    return processor

if __name__ == "__main__":
    # Test the advanced feature system
    processor = create_production_dataset()
    
    # Get training data
    train_features, train_labels, test_features, test_labels = processor.get_training_data()
    
    print(f"\n✅ Training Data Ready:")
    print(f"Training samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print(f"Feature count: {len([k for k in train_features.keys() if 'odds' not in k])}")
    
    # Show some advanced features in action
    print(f"\n🔍 Sample Advanced Feature Values:")
    advanced_keys = [k for k in train_features.keys() if any(x in k for x in 
                    ['ppg-', 'gd-trend', 'big6', 'market-', 'pressure', 'streak'])][:10]
    
    for key in advanced_keys:
        values = train_features[key][:5]
        print(f"{key}: [{', '.join([f'{v:.3f}' for v in values])}]")