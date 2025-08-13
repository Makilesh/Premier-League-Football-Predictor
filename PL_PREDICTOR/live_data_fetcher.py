"""
LIVE DATA FETCHER FOR PREMIER LEAGUE BETTING SYSTEM
==================================================

This module provides live data integration with football-data.org API
and converts the data to match our 180+ feature format for predictions.

Classes:
- FootballDataFetcher: API integration with rate limiting and caching
- LiveBettingIntegration: Feature transformation and prediction pipeline

Author: Premier League Prediction System
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import pickle
from collections import defaultdict
import sys

# Import our existing components
from enhanced_dataset import MatchResult, EnhancedDataset
from advanced_features import AdvancedFeatureEngine
from production_betting_system import ProductionBettingSystem

@dataclass
class LiveFixture:
    """Structure for live fixture data from API"""
    id: int
    utc_date: str
    status: str
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    odds: Optional[Dict] = None
    
class FootballDataFetcher:
    """
    Professional football data fetcher with rate limiting and caching
    Connects to football-data.org API for live Premier League data
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.football-data.org/v4"
        self.rate_limit_delay = 6.1  # 10 requests per minute = 6+ second delay
        self.last_request_time = 0
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        
        # Premier League competition ID
        self.pl_competition_id = 2021
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Team name mappings to match historical data
        self.team_mappings = {
            'Arsenal FC': 'Arsenal',
            'Arsenal': 'Arsenal',
            'Chelsea FC': 'Chelsea',
            'Chelsea': 'Chelsea', 
            'Liverpool FC': 'Liverpool',
            'Liverpool': 'Liverpool',
            'Manchester City FC': 'Man City',
            'Manchester City': 'Man City',
            'Manchester United FC': 'Man United',
            'Manchester United': 'Man United',
            'Tottenham Hotspur FC': 'Tottenham',
            'Tottenham': 'Tottenham',
            'Brighton & Hove Albion FC': 'Brighton',
            'Brighton': 'Brighton',
            'Crystal Palace FC': 'Crystal Palace',
            'Crystal Palace': 'Crystal Palace',
            'Everton FC': 'Everton',
            'Everton': 'Everton',
            'Leeds United FC': 'Leeds',
            'Leicester City FC': 'Leicester',
            'Newcastle United FC': 'Newcastle',
            'Newcastle': 'Newcastle',
            'Norwich City FC': 'Norwich',
            'Southampton FC': 'Southampton',
            'Watford FC': 'Watford',
            'West Ham United FC': 'West Ham',
            'West Ham': 'West Ham',
            'Wolverhampton Wanderers FC': 'Wolves',
            'Wolves': 'Wolves',
            'Burnley FC': 'Burnley',
            'Burnley': 'Burnley',
            'Aston Villa FC': 'Aston Villa',
            'Aston Villa': 'Aston Villa',
            'Brentford FC': 'Brentford',
            'AFC Bournemouth': 'Bournemouth',
            'Fulham FC': 'Fulham',
            'Fulham': 'Fulham',
            'Nottingham Forest FC': 'Nott\'m Forest',
            'Sunderland AFC': 'Sunderland',
            'Sunderland': 'Sunderland',
            'Sheffield United FC': 'Sheffield United',
            'West Bromwich Albion FC': 'West Brom',
            'West Brom': 'West Brom',
            'Blackburn Rovers FC': 'Blackburn',
            'Blackburn': 'Blackburn',
            'Bolton Wanderers FC': 'Bolton',
            'Bolton': 'Bolton',
            'Stoke City FC': 'Stoke',
            'Hull City AFC': 'Hull',
            'Middlesbrough FC': 'Middlesbrough',
            'Portsmouth FC': 'Portsmouth',
            'Wigan Athletic FC': 'Wigan',
            'Blackpool FC': 'Blackpool',
            'Birmingham City FC': 'Birmingham'
        }
        
        self.logger.info("FootballDataFetcher initialized with API key")
    
    def _rate_limit_wait(self):
        """Ensure we don't exceed API rate limits (10 requests/minute)"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last
            self.logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _make_api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make rate-limited API request with error handling"""
        
        # Check cache first
        cache_key = f"{endpoint}_{str(params)}"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cache_data
        
        # Rate limiting
        self._rate_limit_wait()
        
        url = f"{self.base_url}{endpoint}"
        headers = {'X-Auth-Token': self.api_key}
        
        try:
            self.logger.debug(f"Making API request to: {endpoint}")
            response = requests.get(url, headers=headers, params=params or {})
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the response
            self.cache[cache_key] = (time.time(), data)
            
            return data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode API response: {str(e)}")
            return None
    
    def get_premier_league_fixtures(self, days_ahead: int = 7) -> List[LiveFixture]:
        """Get upcoming Premier League fixtures"""
        
        self.logger.info(f"Fetching Premier League fixtures for next {days_ahead} days...")
        
        # Date range
        date_from = datetime.now().strftime('%Y-%m-%d')
        date_to = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        params = {
            'dateFrom': date_from,
            'dateTo': date_to,
            'status': 'SCHEDULED'
        }
        
        data = self._make_api_request(f"/competitions/{self.pl_competition_id}/matches", params)
        
        if not data or 'matches' not in data:
            self.logger.warning("No fixture data received from API")
            return []
        
        fixtures = []
        
        for match in data['matches']:
            try:
                # Map team names to our historical data format
                home_team = self.team_mappings.get(match['homeTeam']['name'], match['homeTeam']['name'])
                away_team = self.team_mappings.get(match['awayTeam']['name'], match['awayTeam']['name'])
                
                fixture = LiveFixture(
                    id=match['id'],
                    utc_date=match['utcDate'],
                    status=match['status'],
                    home_team=home_team,
                    away_team=away_team,
                    home_team_id=match['homeTeam']['id'],
                    away_team_id=match['awayTeam']['id']
                )
                
                fixtures.append(fixture)
                
            except KeyError as e:
                self.logger.warning(f"Skipping malformed fixture data: {str(e)}")
                continue
        
        self.logger.info(f"Successfully fetched {len(fixtures)} upcoming fixtures")
        return fixtures
    
    def get_team_statistics(self, team_id: int, season: str = "2023") -> Optional[Dict]:
        """Get team statistics for current season"""
        
        endpoint = f"/teams/{team_id}"
        data = self._make_api_request(endpoint)
        
        if data:
            self.logger.debug(f"Retrieved statistics for team ID {team_id}")
            
        return data
    
    def get_odds_data(self, match_id: int) -> Optional[Dict]:
        """Get odds data for a specific match (if available)"""
        
        # Note: football-data.org API doesn't provide odds in free tier
        # This is a placeholder for when odds data becomes available
        # or for integration with other odds providers
        
        self.logger.warning("Odds data not available in current API tier")
        return None
    
    def display_upcoming_matches(self, days_ahead: int = 7):
        """Display upcoming matches in a formatted way"""
        
        fixtures = self.get_premier_league_fixtures(days_ahead)
        
        if not fixtures:
            print("❌ No upcoming fixtures found")
            return
        
        print(f"\nUPCOMING PREMIER LEAGUE FIXTURES (Next {days_ahead} days)")
        print("=" * 60)
        
        for fixture in fixtures:
            match_date = datetime.fromisoformat(fixture.utc_date.replace('Z', '+00:00'))
            formatted_date = match_date.strftime('%Y-%m-%d %H:%M UTC')
            
            print(f"Date: {formatted_date}")
            print(f"Match: {fixture.home_team} vs {fixture.away_team}")
            print(f"Match ID: {fixture.id}")
            print("-" * 40)
    
    def get_head_to_head_data(self, home_team_id: int, away_team_id: int, 
                             limit: int = 10) -> List[Dict]:
        """Get head-to-head match history between two teams"""
        
        endpoint = f"/teams/{home_team_id}/matches"
        params = {'limit': 50}  # Get recent matches to filter
        
        data = self._make_api_request(endpoint, params)
        
        if not data or 'matches' not in data:
            return []
        
        h2h_matches = []
        
        for match in data['matches']:
            if (match['homeTeam']['id'] == away_team_id or 
                match['awayTeam']['id'] == away_team_id):
                
                if match['status'] == 'FINISHED':
                    h2h_matches.append(match)
                    
                    if len(h2h_matches) >= limit:
                        break
        
        return h2h_matches

class LiveBettingIntegration:
    """
    Integration layer that converts live API data to our feature format
    and provides betting recommendations using the trained model
    """
    
    def __init__(self, api_key: str, betting_system: ProductionBettingSystem):
        self.fetcher = FootballDataFetcher(api_key)
        self.betting_system = betting_system
        self.logger = logging.getLogger(__name__)
        
        # Load historical data for feature generation
        try:
            self.historical_dataset = EnhancedDataset('data/combined_all_seasons.csv')
            self.feature_engine = AdvancedFeatureEngine(self.historical_dataset)
        except FileNotFoundError:
            self.logger.error("Historical data file not found. Features will be limited.")
            self.historical_dataset = None
            self.feature_engine = None
        
        # Default odds when not available from API
        self.default_odds = {
            'home': 2.0,
            'draw': 3.3,
            'away': 3.5
        }
        
        self.logger.info("LiveBettingIntegration initialized")
    
    def _map_fixture_to_historical_format(self, fixture: LiveFixture, 
                                        historical_data: List[MatchResult]) -> Optional[MatchResult]:
        """Convert live fixture to historical match result format for feature generation"""
        
        # Find recent matches for both teams to estimate values
        home_matches = [m for m in historical_data 
                       if (m.home_team == fixture.home_team or m.away_team == fixture.home_team)]
        away_matches = [m for m in historical_data
                       if (m.home_team == fixture.away_team or m.away_team == fixture.away_team)]
        
        if not home_matches or not away_matches:
            self.logger.warning(f"Insufficient historical data for {fixture.home_team} vs {fixture.away_team}")
            return None
        
        # Get average stats for estimation
        def get_avg_stats(matches, team):
            if not matches:
                return {'goals': 1.5, 'shots': 12, 'shots_target': 4}
            
            recent = matches[-10:]  # Last 10 matches
            total_goals = total_shots = total_st = 0
            
            for m in recent:
                is_home = m.home_team == team
                if is_home:
                    total_goals += m.home_goals
                    total_shots += m.home_shots
                    total_st += m.home_shots_target
                else:
                    total_goals += m.away_goals
                    total_shots += m.away_shots
                    total_st += m.away_shots_target
            
            return {
                'goals': total_goals / len(recent),
                'shots': total_shots / len(recent),
                'shots_target': total_st / len(recent)
            }
        
        home_stats = get_avg_stats(home_matches, fixture.home_team)
        away_stats = get_avg_stats(away_matches, fixture.away_team)
        
        # Create a pseudo match result for feature generation
        match_date = datetime.fromisoformat(fixture.utc_date.replace('Z', '+00:00'))
        
        pseudo_match = MatchResult(
            date=match_date,
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            result='H',  # Dummy result, not used for prediction
            home_goals=int(home_stats['goals']),
            away_goals=int(away_stats['goals']),
            odds_home=self.default_odds['home'],
            odds_draw=self.default_odds['draw'],
            odds_away=self.default_odds['away'],
            home_shots=int(home_stats['shots']),
            away_shots=int(away_stats['shots']),
            home_shots_target=int(home_stats['shots_target']),
            away_shots_target=int(away_stats['shots_target'])
        )
        
        return pseudo_match
    
    def _generate_features_for_fixture(self, fixture: LiveFixture) -> Optional[Dict]:
        """Generate all 180+ features for a live fixture"""
        
        if not self.historical_dataset or not self.feature_engine:
            self.logger.error("Cannot generate features without historical data")
            return None
        
        # Convert fixture to historical format
        pseudo_match = self._map_fixture_to_historical_format(
            fixture, self.historical_dataset.match_objects
        )
        
        if not pseudo_match:
            return None
        
        try:
            # Ensure pseudo_match date is timezone-naive for comparison
            if hasattr(pseudo_match.date, 'tzinfo') and pseudo_match.date.tzinfo is not None:
                pseudo_match_date = pseudo_match.date.replace(tzinfo=None)
            else:
                pseudo_match_date = pseudo_match.date
                
            # Also ensure fixture date is timezone-naive
            fixture_date = datetime.fromisoformat(fixture.utc_date.replace('Z', '+00:00'))
            if hasattr(fixture_date, 'tzinfo') and fixture_date.tzinfo is not None:
                fixture_date = fixture_date.replace(tzinfo=None)
            
            # Generate enhanced dataset features
            home_stats = self.historical_dataset._get_enhanced_statistics(
                fixture.home_team, fixture_date, is_home=True
            )
            away_stats = self.historical_dataset._get_enhanced_statistics(
                fixture.away_team, fixture_date, is_home=False
            )
            
            if not home_stats or not away_stats:
                self.logger.warning(f"Could not generate basic stats for {fixture.home_team} vs {fixture.away_team}")
                return None
            
            # Head-to-head statistics
            h2h_stats = self.historical_dataset._get_head_to_head_stats(
                fixture.home_team, fixture.away_team, fixture_date
            )
            
            # Rest days calculation
            home_rest = self.historical_dataset._calculate_rest_days(fixture.home_team, fixture_date)
            away_rest = self.historical_dataset._calculate_rest_days(fixture.away_team, fixture_date)
            
            # Momentum calculation
            home_momentum = self.historical_dataset._calculate_momentum(fixture.home_team, fixture_date)
            away_momentum = self.historical_dataset._calculate_momentum(fixture.away_team, fixture_date)
            
            # Generate advanced features
            advanced_features = self.feature_engine._extract_match_features(pseudo_match)
            
            if not advanced_features:
                self.logger.warning("Could not generate advanced features")
                return None
            
            # Combine all features
            features = {
                'odds-home': self.default_odds['home'],
                'odds-draw': self.default_odds['draw'],
                'odds-away': self.default_odds['away'],
                'home-rest-days': home_rest,
                'away-rest-days': away_rest,
                'home-momentum': home_momentum,
                'away-momentum': away_momentum,
                'big-6-home': 1 if fixture.home_team in self.historical_dataset.big_6_teams else 0,
                'big-6-away': 1 if fixture.away_team in self.historical_dataset.big_6_teams else 0,
                'big-6-match': 1 if (fixture.home_team in self.historical_dataset.big_6_teams and 
                                   fixture.away_team in self.historical_dataset.big_6_teams) else 0,
            }
            
            # Add team statistics with prefixes
            for prefix, stats in [('home', home_stats), ('away', away_stats)]:
                for key, value in stats.items():
                    features[f'{prefix}-{key}'] = value
            
            # Add head-to-head statistics
            for key, value in h2h_stats.items():
                features[f'h2h-{key}'] = value
            
            # Add advanced features (skip duplicates and result)
            for key, value in advanced_features.items():
                if key not in features and key != 'result':
                    features[key] = value
            
            self.logger.debug(f"Generated {len(features)} features for {fixture.home_team} vs {fixture.away_team}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating features: {str(e)}")
            return None
    
    def get_live_recommendations(self, days_ahead: int = 7) -> str:
        """Get live betting recommendations for upcoming matches"""
        
        try:
            self.logger.info(f"Generating live recommendations for next {days_ahead} days...")
            
            # Get upcoming fixtures
            fixtures = self.fetcher.get_premier_league_fixtures(days_ahead)
            
            if not fixtures:
                return "❌ No upcoming fixtures found for the specified period."
            
            # Generate features for each fixture
            match_features = {}
            valid_fixtures = []
            
            for i, fixture in enumerate(fixtures):
                features = self._generate_features_for_fixture(fixture)
                
                if features:
                    # Convert to the format expected by the model
                    for key, value in features.items():
                        if key not in match_features:
                            match_features[key] = []
                        match_features[key].append(float(value))
                    
                    valid_fixtures.append(fixture)
                else:
                    self.logger.warning(f"Skipping fixture {fixture.home_team} vs {fixture.away_team} - insufficient data")
            
            if not valid_fixtures:
                return "❌ No valid fixtures found with sufficient historical data."
            
            # Convert to numpy arrays
            for key in match_features:
                match_features[key] = np.array(match_features[key], dtype=np.float32)
            
            # Get recommendations from the betting system
            recommendations = self.betting_system.get_recommendations(match_features)
            
            if not recommendations:
                return "No profitable betting opportunities found in upcoming matches."
            
            # Format recommendations report
            report = f"LIVE BETTING RECOMMENDATIONS\n"
            report += "=" * 50 + "\n"
            report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += f"Analyzed: {len(valid_fixtures)} fixtures\n"
            report += f"Opportunities: {len(recommendations)} matches with value bets\n\n"
            
            for i, match_rec in enumerate(recommendations):
                fixture = valid_fixtures[match_rec['match_index']]
                match_date = datetime.fromisoformat(fixture.utc_date.replace('Z', '+00:00'))
                
                report += f"MATCH {i+1}: {fixture.home_team} vs {fixture.away_team}\n"
                report += f"Date: {match_date.strftime('%Y-%m-%d %H:%M UTC')}\n"
                
                for rec in match_rec['recommendations']:
                    outcome_names = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
                    outcome_emojis = {'H': '🏠', 'D': '🤝', 'A': '✈️'}
                    
                    report += f"\n  {outcome_emojis[rec['outcome']]} {outcome_names[rec['outcome']]}:\n"
                    report += f"     Model Probability: {rec['model_probability']:.1%}\n"
                    report += f"     💹 Market Odds: {rec['market_odds']:.2f}\n"
                    report += f"     📈 Value Edge: {rec['edge']:.1%}\n"
                    report += f"     🎪 Confidence Score: {rec['confidence']:.3f}\n"
                    report += f"     Recommended Bet: £{rec['recommended_bet']:.2f}\n"
                    report += f"     📊 Kelly Fraction: {rec['kelly_fraction']:.1%}\n"
                
                report += "\n" + "-" * 50 + "\n"
            
            # Add disclaimer
            report += "\n⚠️  IMPORTANT NOTES:\n"
            report += "• Recommendations based on historical data and model predictions\n"
            report += "• Always verify current odds before placing bets\n"
            report += "• Consider current team news and injuries\n"
            report += "• Bet responsibly and within your means\n"
            report += f"• Model accuracy: ~{self.betting_system.ensemble.model_weights.get('rf', 0) * 100:.0f}% (ensemble)\n"
            
            return report
            
        except Exception as e:
            error_msg = f"❌ Error generating recommendations: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def analyze_specific_match(self, home_team: str, away_team: str, 
                              match_date: str = None) -> str:
        """Analyze a specific match and provide detailed breakdown"""
        
        try:
            # Create a dummy fixture for the specified match
            if match_date:
                fixture_date = datetime.fromisoformat(match_date)
            else:
                fixture_date = datetime.now() + timedelta(days=1)
            
            fixture = LiveFixture(
                id=999999,  # Dummy ID
                utc_date=fixture_date.isoformat() + 'Z',
                status='SCHEDULED',
                home_team=home_team,
                away_team=away_team,
                home_team_id=1,  # Dummy ID
                away_team_id=2   # Dummy ID
            )
            
            features = self._generate_features_for_fixture(fixture)
            
            if not features:
                return f"❌ Unable to analyze {home_team} vs {away_team} - insufficient historical data"
            
            # Convert to model format
            match_features = {}
            for key, value in features.items():
                match_features[key] = np.array([float(value)], dtype=np.float32)
            
            # Get prediction
            predictions = self.betting_system.ensemble.predict_with_confidence(match_features)
            prediction = predictions[0]
            
            # Format detailed analysis
            report = f"🔍 DETAILED MATCH ANALYSIS\n"
            report += "=" * 40 + "\n"
            report += f"MATCH: {home_team} vs {away_team}\n"
            report += f"Date: {fixture_date.strftime('%Y-%m-%d')}\n\n"
            
            report += "MODEL PREDICTIONS:\n"
            report += f"  🏠 Home Win: {prediction['probabilities']['H']:.1%}\n"
            report += f"  🤝 Draw: {prediction['probabilities']['D']:.1%}\n"
            report += f"  ✈️  Away Win: {prediction['probabilities']['A']:.1%}\n"
            report += f"  📊 Most Likely: {prediction['prediction']}\n"
            report += f"  🎪 Confidence: {prediction['confidence']:.3f}\n\n"
            
            report += "📈 KEY FEATURES ANALYSIS:\n"
            
            # Show some key features
            key_features = [
                ('home-ppg-5', 'Home Points/Game (L5)'),
                ('away-ppg-5', 'Away Points/Game (L5)'),
                ('home-gd-5', 'Home Goal Diff (L5)'),
                ('away-gd-5', 'Away Goal Diff (L5)'),
                ('big-6-match', 'Big 6 Clash'),
                ('home-rest-days', 'Home Rest Days'),
                ('away-rest-days', 'Away Rest Days')
            ]
            
            for feature, description in key_features:
                if feature in features:
                    value = features[feature]
                    if isinstance(value, (int, float)):
                        report += f"  • {description}: {value:.2f}\n"
                    else:
                        report += f"  • {description}: {value}\n"
            
            return report
            
        except Exception as e:
            return f"❌ Error analyzing match: {str(e)}"
    
    def get_team_form_analysis(self, team_name: str, matches: int = 10) -> str:
        """Get detailed form analysis for a specific team"""
        
        if not self.historical_dataset:
            return "❌ Historical data not available"
        
        team_matches = [m for m in self.historical_dataset.match_objects 
                       if (m.home_team == team_name or m.away_team == team_name)]
        
        if len(team_matches) < matches:
            return f"❌ Insufficient data for {team_name} (need {matches}, have {len(team_matches)})"
        
        recent_matches = team_matches[-matches:]
        
        # Calculate form metrics
        points = goals_for = goals_against = 0
        wins = draws = losses = 0
        home_games = 0
        
        for match in recent_matches:
            is_home = match.home_team == team_name
            
            if is_home:
                team_goals = match.home_goals
                opp_goals = match.away_goals
                home_games += 1
            else:
                team_goals = match.away_goals
                opp_goals = match.home_goals
            
            goals_for += team_goals
            goals_against += opp_goals
            
            # Result from team perspective
            if (is_home and match.result == 'H') or (not is_home and match.result == 'A'):
                wins += 1
                points += 3
            elif match.result == 'D':
                draws += 1
                points += 1
            else:
                losses += 1
        
        # Format report
        report = f"📊 TEAM FORM ANALYSIS: {team_name}\n"
        report += "=" * 40 + "\n"
        report += f"📈 Last {matches} matches:\n\n"
        
        report += f"🏆 RESULTS:\n"
        report += f"  Wins: {wins} | Draws: {draws} | Losses: {losses}\n"
        report += f"  Win Rate: {wins/matches:.1%}\n"
        report += f"  Points: {points}/{matches*3} ({points/matches:.2f} PPG)\n\n"
        
        report += f"GOALS:\n"
        report += f"  Goals For: {goals_for} ({goals_for/matches:.1f} per game)\n"
        report += f"  Goals Against: {goals_against} ({goals_against/matches:.1f} per game)\n"
        report += f"  Goal Difference: {goals_for - goals_against:+d}\n\n"
        
        report += f"🏠 HOME/AWAY:\n"
        report += f"  Home Games: {home_games}/{matches} ({home_games/matches:.1%})\n"
        report += f"  Away Games: {matches-home_games}/{matches} ({(matches-home_games)/matches:.1%})\n\n"
        
        # Recent results
        report += f"RECENT RESULTS:\n"
        for i, match in enumerate(recent_matches[-5:], 1):
            is_home = match.home_team == team_name
            opponent = match.away_team if is_home else match.home_team
            location = "vs" if is_home else "@"
            
            if (is_home and match.result == 'H') or (not is_home and match.result == 'A'):
                result = "W"
            elif match.result == 'D':
                result = "D"
            else:
                result = "L"
            
            if is_home:
                score = f"{match.home_goals}-{match.away_goals}"
            else:
                score = f"{match.away_goals}-{match.home_goals}"
            
            report += f"  {i}. {result} {score} {location} {opponent}\n"
        
        return report

# Utility functions for system integration

def test_api_connection(api_key: str) -> bool:
    """Test API connection and authentication"""
    
    fetcher = FootballDataFetcher(api_key)
    
    try:
        # Try to fetch competitions to test connection
        data = fetcher._make_api_request("/competitions")
        
        if data:
            print("✅ API connection successful")
            print(f"📡 Available competitions: {len(data.get('competitions', []))}")
            return True
        else:
            print("❌ API connection failed")
            return False
            
    except Exception as e:
        print(f"❌ API test error: {str(e)}")
        return False

def validate_team_names(api_key: str) -> Dict[str, str]:
    """Get current Premier League team names for validation"""
    
    fetcher = FootballDataFetcher(api_key)
    fixtures = fetcher.get_premier_league_fixtures(30)  # Next 30 days
    
    teams = set()
    for fixture in fixtures:
        teams.add(fixture.home_team)
        teams.add(fixture.away_team)
    
    return {team: team for team in sorted(teams)}

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('live_betting.log')
        ]
    )
    
    # Test with environment variable
    api_key = os.getenv('FOOTBALL_DATA_API_KEY')
    
    if not api_key:
        print("❌ Please set FOOTBALL_DATA_API_KEY environment variable")
        sys.exit(1)
    
    print("🔧 Testing Live Data Fetcher...")
    
    # Test API connection
    if not test_api_connection(api_key):
        sys.exit(1)
    
    # Test fetcher
    fetcher = FootballDataFetcher(api_key)
    fetcher.display_upcoming_matches(7)
    
    # Test team validation
    teams = validate_team_names(api_key)
    print(f"\n📋 Current Premier League teams: {len(teams)}")
    for team in list(teams.keys())[:5]:  # Show first 5
        print(f"  • {team}")
    
    print("\n✅ Live Data Fetcher ready for integration!")
