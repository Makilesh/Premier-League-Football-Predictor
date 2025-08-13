#!/usr/bin/env python3
"""
QUICK MATCH PREDICTIONS SCRIPT
=============================

This script bypasses the complex timezone issues and gives you direct predictions
"""

import os
import sys
from datetime import datetime
from live_data_fetcher import FootballDataFetcher

def get_simple_predictions():
    """Get simple match predictions without complex historical integration"""
    
    print("PREMIER LEAGUE MATCH PREDICTIONS")
    print("=" * 50)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize API fetcher
    api_key = "476a7d56f2d0423586f085220bb86858"
    fetcher = FootballDataFetcher(api_key)
    
    # Get upcoming fixtures
    print("Fetching upcoming matches...")
    fixtures = fetcher.get_premier_league_fixtures(7)
    
    if not fixtures:
        print("❌ No fixtures found")
        return
    
    print(f"✅ Found {len(fixtures)} upcoming matches")
    print()
    
    # Simple predictions based on team strength
    team_strengths = {
        'Liverpool': 95, 'Manchester City': 94, 'Arsenal': 88, 'Manchester United': 82,
        'Chelsea': 85, 'Tottenham': 80, 'Newcastle': 78, 'Brighton': 72,
        'Aston Villa': 75, 'West Ham': 68, 'Crystal Palace': 65, 'Fulham': 70,
        'Wolves': 62, 'Everton': 60, 'Bournemouth': 58, 'Brentford': 63,
        'Burnley': 45, 'Leeds': 50, 'Sunderland': 55, "Nott'm Forest": 62
    }
    
    for i, fixture in enumerate(fixtures, 1):
        home_team = fixture.home_team
        away_team = fixture.away_team
        match_date = datetime.fromisoformat(fixture.utc_date.replace('Z', '+00:00'))
        
        # Calculate probabilities based on team strength
        home_strength = team_strengths.get(home_team, 60)
        away_strength = team_strengths.get(away_team, 60)
        
        # Home advantage (+5 points)
        home_strength += 5
        
        # Calculate probabilities
        total_strength = home_strength + away_strength
        home_prob = (home_strength / total_strength) * 0.7  # Scale to realistic range
        away_prob = (away_strength / total_strength) * 0.5
        draw_prob = 1.0 - home_prob - away_prob
        
        # Normalize if needed
        if draw_prob < 0.15:
            draw_prob = 0.15
            remaining = 0.85
            home_prob = (home_prob / (home_prob + away_prob)) * remaining
            away_prob = remaining - home_prob
        
        # Determine prediction
        if home_prob > away_prob and home_prob > draw_prob:
            prediction = f"{home_team} Win"
            confidence = home_prob
        elif away_prob > home_prob and away_prob > draw_prob:
            prediction = f"{away_team} Win" 
            confidence = away_prob
        else:
            prediction = "Draw"
            confidence = draw_prob
            
        # Format output
        print(f"🏈 MATCH {i}: {home_team} vs {away_team}")
        print(f"📅 Date: {match_date.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"🎯 PREDICTION: {prediction} ({confidence:.1%} confidence)")
        print("📊 PROBABILITIES:")
        print(f"   • Home Win ({home_team}): {home_prob:.1%}")
        print(f"   • Draw: {draw_prob:.1%}")  
        print(f"   • Away Win ({away_team}): {away_prob:.1%}")
        print()
        print("💰 BETTING SUGGESTION:")
        if confidence > 0.55:
            print(f"   💚 STRONG BET: {prediction}")
        elif confidence > 0.45:
            print(f"   💛 MODERATE BET: {prediction}")
        else:
            print(f"   🔸 WEAK BET: Too close to call")
        print("-" * 60)
        print()

if __name__ == "__main__":
    get_simple_predictions()
