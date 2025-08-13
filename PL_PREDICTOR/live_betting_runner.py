"""
COMPLETE LIVE PREMIER LEAGUE BETTING SYSTEM
==========================================

This script combines your trained betting system with live data from football-data.org
to provide real-time betting recommendations.

Setup Instructions:
1. Get your API key from https://www.football-data.org/client/register
2. Set environment variable: export FOOTBALL_DATA_API_KEY='476a7d56f2d0423586f085220bb86858'
3. Ensure you have your historical data file: data/book.csv
4. Run this script to get live recommendations
"""

import os
import sys
from datetime import datetime
import logging
from typing import Optional

# Import our components
from live_data_fetcher import FootballDataFetcher, LiveBettingIntegration
from production_betting_system import ProductionBettingSystem, PerformanceDashboard

class LiveBettingRunner:
    """
    Main runner for the complete live betting system
    """
    
    def __init__(self, api_key: str, historical_data_path: str = 'data/combined_all_seasons.csv', 
                 initial_bankroll: float = 1000):
        
        self.api_key = api_key
        self.historical_data_path = historical_data_path
        self.initial_bankroll = initial_bankroll
        
        # Initialize components
        self.betting_system = None
        self.live_fetcher = None
        self.integration = None
        self.dashboard = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('betting_system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_system(self) -> bool:
        """Initialize and train the complete betting system"""
        
        try:
            self.logger.info("🚀 Initializing Complete Betting System...")
            
            # Check if historical data exists
            if not os.path.exists(self.historical_data_path):
                self.logger.error(f"❌ Historical data file not found: {self.historical_data_path}")
                return False
            
            # Initialize betting system
            self.logger.info("📊 Loading historical data and training models...")
            self.betting_system = ProductionBettingSystem(
                self.historical_data_path, 
                self.initial_bankroll
            )
            
            # Train the system
            training_results = self.betting_system.train_system(train_fraction=0.85)
            
            # Initialize live data fetcher
            self.logger.info("🌐 Initializing live data fetcher...")
            self.live_fetcher = FootballDataFetcher(self.api_key)
            
            # Create integration layer
            self.integration = LiveBettingIntegration(self.api_key, self.betting_system)
            
            # Initialize dashboard
            self.dashboard = PerformanceDashboard(self.betting_system)
            
            # Log training results
            self.logger.info("System initialized successfully!")
            self.logger.info(f"Model Accuracy: {training_results['backtest_results']['accuracy']:.3f}")
            self.logger.info(f"Backtest ROI: {training_results['backtest_results']['betting_performance'].get('roi', 0):.1f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize system: {str(e)}")
            return False
    
    def get_live_recommendations(self, days_ahead: int = 7) -> str:
        """Get live betting recommendations"""
        
        if not self.integration:
            return "❌ System not initialized. Run initialize_system() first."
        
        self.logger.info(f"Getting live recommendations for next {days_ahead} days...")
        
        try:
            # Get live recommendations
            recommendations = self.integration.get_live_recommendations(days_ahead)
            
            # Log the activity
            self.logger.info("Live recommendations generated successfully")
            
            return recommendations
            
        except Exception as e:
            error_msg = f"❌ Error getting live recommendations: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def show_upcoming_matches(self, days_ahead: int = 7):
        """Display upcoming matches"""
        
        if not self.live_fetcher:
            print("❌ System not initialized.")
            return
        
        self.live_fetcher.display_upcoming_matches(days_ahead)
    
    def show_system_performance(self) -> str:
        """Show current system performance"""
        
        if not self.dashboard:
            return "❌ System not initialized."
        
        return self.dashboard.generate_performance_report()
    
    def run_daily_analysis(self):
        """Run daily betting analysis - suitable for automation"""
        
        self.logger.info("Running daily betting analysis...")
        
        print("\n" + "=" * 60)
        print("DAILY PREMIER LEAGUE BETTING ANALYSIS")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show upcoming matches
        print("\nUPCOMING MATCHES:")
        self.show_upcoming_matches(7)
        
        # Get recommendations
        print("\nBETTING RECOMMENDATIONS:")
        recommendations = self.get_live_recommendations(7)
        print(recommendations)

# Main execution
if __name__ == "__main__":
    import os
    
    # Set API key
    os.environ['FOOTBALL_DATA_API_KEY'] = "476a7d56f2d0423586f085220bb86858"
    
    # Create and run the system
    runner = LiveBettingRunner(
        api_key="476a7d56f2d0423586f085220bb86858",
        historical_data_path='data/combined_all_seasons.csv',
        initial_bankroll=1000
    )
    
    print("Initializing Premier League Betting System...")
    if runner.initialize_system():
        print("System initialized successfully!")
        print("\nRunning daily analysis...")
        runner.run_daily_analysis()
    else:
        print("Failed to initialize system")