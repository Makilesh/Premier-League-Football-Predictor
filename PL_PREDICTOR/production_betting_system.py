import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import logging
import json
import pickle
from datetime import datetime
from advanced_features import AdvancedDatasetProcessor
import warnings
warnings.filterwarnings('ignore')

class ModelEnsemble:
    """
    Advanced model ensemble combining Random Forest, Gradient Boosting, and Neural Network
    with confidence scoring and dynamic feature weighting
    """
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = [f for f in feature_names if 'odds' not in f]
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_weights = {'rf': 0.33, 'gb': 0.33, 'nn': 0.34}
        self.is_trained = False
        
    def initialize_models(self):
        """Initialize all models with optimized parameters"""
        
        # Random Forest - Good for feature interactions
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting - Good for sequential patterns
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Neural Network - Good for complex non-linear patterns
        self.models['nn'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # Scalers for neural network
        self.scalers['nn'] = StandardScaler()
    
    def train(self, X: Dict, y: np.ndarray) -> Dict:
        """Train all models and calculate ensemble weights"""
        
        logging.info("Training model ensemble...")
        
        # Convert features dict to array
        X_array = self._dict_to_array(X)
        
        # Initialize models
        self.initialize_models()
        
        training_results = {}
        
        # Train Random Forest
        logging.info("Training Random Forest...")
        self.models['rf'].fit(X_array, y)
        rf_pred = self.models['rf'].predict(X_array)
        rf_accuracy = accuracy_score(y, rf_pred)
        training_results['rf'] = {'accuracy': rf_accuracy}
        
        # Store feature importance
        self.feature_importance['rf'] = dict(zip(
            self.feature_names, self.models['rf'].feature_importances_
        ))
        
        # Train Gradient Boosting
        logging.info("Training Gradient Boosting...")
        self.models['gb'].fit(X_array, y)
        gb_pred = self.models['gb'].predict(X_array)
        gb_accuracy = accuracy_score(y, gb_pred)
        training_results['gb'] = {'accuracy': gb_accuracy}
        
        # Store feature importance
        self.feature_importance['gb'] = dict(zip(
            self.feature_names, self.models['gb'].feature_importances_
        ))
        
        # Train Neural Network
        logging.info("Training Neural Network...")
        X_scaled = self.scalers['nn'].fit_transform(X_array)
        self.models['nn'].fit(X_scaled, y)
        nn_pred = self.models['nn'].predict(X_scaled)
        nn_accuracy = accuracy_score(y, nn_pred)
        training_results['nn'] = {'accuracy': nn_accuracy}
        
        # Update model weights based on performance
        total_accuracy = rf_accuracy + gb_accuracy + nn_accuracy
        self.model_weights['rf'] = rf_accuracy / total_accuracy
        self.model_weights['gb'] = gb_accuracy / total_accuracy
        self.model_weights['nn'] = nn_accuracy / total_accuracy
        
        self.is_trained = True
        
        logging.info(f"Ensemble trained - RF: {rf_accuracy:.3f}, GB: {gb_accuracy:.3f}, NN: {nn_accuracy:.3f}")
        
        return training_results
    
    def predict_proba(self, X: Dict) -> np.ndarray:
        """Get ensemble probability predictions"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        X_array = self._dict_to_array(X)
        
        # Get predictions from each model
        rf_proba = self.models['rf'].predict_proba(X_array)
        gb_proba = self.models['gb'].predict_proba(X_array)
        
        X_scaled = self.scalers['nn'].transform(X_array)
        nn_proba = self.models['nn'].predict_proba(X_scaled)
        
        # Weighted ensemble
        ensemble_proba = (
            rf_proba * self.model_weights['rf'] +
            gb_proba * self.model_weights['gb'] +
            nn_proba * self.model_weights['nn']
        )
        
        return ensemble_proba
    
    def predict_with_confidence(self, X: Dict) -> List[Dict]:
        """Get predictions with confidence scores"""
        
        probabilities = self.predict_proba(X)
        predictions = []
        
        for probs in probabilities:
            max_prob_idx = np.argmax(probs)
            max_prob = probs[max_prob_idx]
            
            # Calculate confidence based on probability margin
            sorted_probs = np.sort(probs)[::-1]
            confidence = sorted_probs[0] - sorted_probs[1]  # Margin between 1st and 2nd
            
            predictions.append({
                'prediction': ['H', 'D', 'A'][max_prob_idx],
                'probabilities': {'H': probs[0], 'D': probs[1], 'A': probs[2]},
                'confidence': confidence,
                'max_probability': max_prob
            })
        
        return predictions
    
    def _dict_to_array(self, X: Dict) -> np.ndarray:
        """Convert feature dictionary to numpy array"""
        return np.column_stack([X[feature] for feature in self.feature_names])
    
    def get_top_features(self, n: int = 20) -> Dict:
        """Get top N most important features from each model"""
        
        if not self.feature_importance:
            return {}
        
        top_features = {}
        
        for model_name, importance in self.feature_importance.items():
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_features[model_name] = sorted_features[:n]
        
        return top_features

class KellyCriterionBetting:
    """
    Kelly Criterion implementation for optimal bet sizing
    """
    
    def __init__(self, bankroll: float, max_bet_fraction: float = 0.05):
        self.initial_bankroll = bankroll
        self.current_bankroll = bankroll
        self.max_bet_fraction = max_bet_fraction
        self.bet_history = []
    
    def calculate_kelly_fraction(self, probability: float, odds: float) -> float:
        """Calculate Kelly fraction for a bet"""
        
        if probability <= 0 or odds <= 1:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds-1, p = probability, q = 1-p
        b = odds - 1
        q = 1 - probability
        
        kelly_fraction = (b * probability - q) / b
        
        # Cap at maximum bet fraction for safety
        return max(0, min(kelly_fraction, self.max_bet_fraction))
    
    def calculate_bet_size(self, probability: float, odds: float) -> float:
        """Calculate optimal bet size"""
        
        kelly_fraction = self.calculate_kelly_fraction(probability, odds)
        bet_size = self.current_bankroll * kelly_fraction
        
        return bet_size
    
    def place_bet(self, bet_size: float, odds: float, outcome: str, 
                  actual_result: str, bet_type: str) -> Dict:
        """Place a bet and update bankroll"""
        
        won = (outcome == actual_result)
        
        if won:
            profit = bet_size * (odds - 1)
            self.current_bankroll += profit
        else:
            self.current_bankroll -= bet_size
        
        bet_record = {
            'bet_size': bet_size,
            'odds': odds,
            'outcome': outcome,
            'actual_result': actual_result,
            'bet_type': bet_type,
            'won': won,
            'profit': profit if won else -bet_size,
            'bankroll_after': self.current_bankroll,
            'timestamp': datetime.now()
        }
        
        self.bet_history.append(bet_record)
        
        return bet_record
    
    def get_performance_stats(self) -> Dict:
        """Calculate betting performance statistics"""
        
        if not self.bet_history:
            return {}
        
        total_bets = len(self.bet_history)
        wins = sum(1 for bet in self.bet_history if bet['won'])
        total_profit = sum(bet['profit'] for bet in self.bet_history)
        total_staked = sum(bet['bet_size'] for bet in self.bet_history)
        
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        bankroll_growth = ((self.current_bankroll - self.initial_bankroll) / 
                          self.initial_bankroll * 100)
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': total_bets - wins,
            'win_rate': wins / total_bets * 100,
            'total_profit': total_profit,
            'total_staked': total_staked,
            'roi': roi,
            'bankroll_growth': bankroll_growth,
            'current_bankroll': self.current_bankroll
        }

class ProductionBettingSystem:
    """
    Complete production-ready betting system
    """
    
    def __init__(self, data_file: str, initial_bankroll: float = 1000):
        
        logging.basicConfig(level=logging.INFO)
        
        # Load and process data
        logging.info("Loading advanced dataset...")
        self.data_processor = AdvancedDatasetProcessor(data_file)
        
        # Initialize components
        self.ensemble = None
        self.kelly_betting = KellyCriterionBetting(initial_bankroll)
        
        # Strategy parameters
        self.betting_config = {
            'min_edge': {'H': 0.02, 'D': 0.03, 'A': 0.02},
            'min_confidence': 0.15,
            'min_probability': 0.3,
            'max_probability': 0.8
        }
        
    def train_system(self, train_fraction: float = 0.8) -> Dict:
        """Train the complete betting system"""
        
        # Get training data
        train_X, train_y, test_X, test_y = self.data_processor.get_training_data(train_fraction)
        
        # Initialize and train ensemble
        feature_names = [k for k in train_X.keys() if 'odds' not in k]
        self.ensemble = ModelEnsemble(feature_names)
        
        training_results = self.ensemble.train(train_X, train_y)
        
        # Test system performance
        test_results = self.backtest_system(test_X, test_y)
        
        return {
            'training_results': training_results,
            'backtest_results': test_results,
            'model_weights': self.ensemble.model_weights,
            'feature_count': len(feature_names)
        }
    
    def backtest_system(self, test_X: Dict, test_y: np.ndarray) -> Dict:
        """Comprehensive backtesting of the betting system"""
        
        if not self.ensemble or not self.ensemble.is_trained:
            raise ValueError("System must be trained before backtesting")
        
        logging.info("Running backtest...")
        
        # Reset Kelly betting for backtest
        initial_bankroll = self.kelly_betting.initial_bankroll
        self.kelly_betting = KellyCriterionBetting(initial_bankroll)
        
        # Get predictions
        predictions = self.ensemble.predict_with_confidence(test_X)
        
        backtest_stats = {
            'total_matches': len(test_y),
            'accuracy': 0,
            'bets_placed': 0,
            'betting_performance': {}
        }
        
        # Calculate accuracy
        ensemble_preds = [pred['prediction'] for pred in predictions]
        accuracy = accuracy_score(test_y, ensemble_preds)
        backtest_stats['accuracy'] = accuracy
        
        # Betting simulation
        outcomes = ['H', 'D', 'A']
        odds_keys = ['odds-home', 'odds-draw', 'odds-away']
        
        for i, prediction in enumerate(predictions):
            actual_result = test_y[i]
            
            for j, outcome in enumerate(outcomes):
                model_prob = prediction['probabilities'][outcome]
                market_odds = test_X[odds_keys[j]][i]
                
                if market_odds <= 1:  # Skip invalid odds
                    continue
                
                implied_prob = 1 / market_odds
                edge = model_prob - implied_prob
                confidence = prediction['confidence']
                
                # Check betting criteria
                if (edge > self.betting_config['min_edge'][outcome] and
                    confidence > self.betting_config['min_confidence'] and
                    self.betting_config['min_probability'] <= model_prob <= self.betting_config['max_probability']):
                    
                    # Calculate bet size using Kelly
                    bet_size = self.kelly_betting.calculate_bet_size(model_prob, market_odds)
                    
                    if bet_size > 0:
                        # Place bet
                        self.kelly_betting.place_bet(
                            bet_size, market_odds, outcome, actual_result, f'{outcome}_bet'
                        )
                        backtest_stats['bets_placed'] += 1
        
        # Get final performance stats
        backtest_stats['betting_performance'] = self.kelly_betting.get_performance_stats()
        
        return backtest_stats
    
    def get_recommendations(self, upcoming_matches: Dict) -> List[Dict]:
        """Get betting recommendations for upcoming matches"""
        
        if not self.ensemble or not self.ensemble.is_trained:
            raise ValueError("System must be trained before making recommendations")
        
        predictions = self.ensemble.predict_with_confidence(upcoming_matches)
        recommendations = []
        
        outcomes = ['H', 'D', 'A']
        odds_keys = ['odds-home', 'odds-draw', 'odds-away']
        
        for i, prediction in enumerate(predictions):
            match_recs = []
            
            for j, outcome in enumerate(outcomes):
                model_prob = prediction['probabilities'][outcome]
                
                # Skip if no odds available
                if odds_keys[j] not in upcoming_matches:
                    continue
                
                market_odds = upcoming_matches[odds_keys[j]][i]
                
                if market_odds <= 1:
                    continue
                
                implied_prob = 1 / market_odds
                edge = model_prob - implied_prob
                confidence = prediction['confidence']
                
                # Check betting criteria
                if (edge > self.betting_config['min_edge'][outcome] and
                    confidence > self.betting_config['min_confidence'] and
                    self.betting_config['min_probability'] <= model_prob <= self.betting_config['max_probability']):
                    
                    bet_size = self.kelly_betting.calculate_bet_size(model_prob, market_odds)
                    
                    if bet_size > 0:
                        match_recs.append({
                            'outcome': outcome,
                            'model_probability': model_prob,
                            'market_odds': market_odds,
                            'edge': edge,
                            'confidence': confidence,
                            'recommended_bet': bet_size,
                            'kelly_fraction': bet_size / self.kelly_betting.current_bankroll
                        })
            
            if match_recs:
                recommendations.append({
                    'match_index': i,
                    'recommendations': match_recs
                })
        
        return recommendations
    
    def save_system(self, filepath: str):
        """Save trained system to disk"""
        
        system_data = {
            'ensemble_models': self.ensemble.models if self.ensemble else None,
            'ensemble_scalers': self.ensemble.scalers if self.ensemble else None,
            'ensemble_weights': self.ensemble.model_weights if self.ensemble else None,
            'feature_names': self.ensemble.feature_names if self.ensemble else None,
            'betting_config': self.betting_config,
            'kelly_settings': {
                'initial_bankroll': self.kelly_betting.initial_bankroll,
                'max_bet_fraction': self.kelly_betting.max_bet_fraction
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(system_data, f)
        
        logging.info(f"System saved to {filepath}")
    
    def get_feature_analysis(self) -> Dict:
        """Get comprehensive feature analysis"""
        
        if not self.ensemble:
            return {}
        
        return {
            'top_features': self.ensemble.get_top_features(15),
            'model_weights': self.ensemble.model_weights,
            'feature_categories': self.data_processor.get_feature_importance_categories()
        }

def run_complete_system(data_file: str = 'data/combined_all_seasons.csv', bankroll: float = 1000):
    """Run the complete production betting system"""
    
    print("🚀 Initializing Production Betting System")
    print("=" * 60)
    
    # Initialize system
    system = ProductionBettingSystem(data_file, bankroll)
    
    # Train system
    print("\n🎯 Training System...")
    results = system.train_system(train_fraction=0.8)
    
    # Print results
    print(f"\n📊 Training Results:")
    for model, stats in results['training_results'].items():
        print(f"  {model.upper()}: {stats['accuracy']:.3f} accuracy")
    
    print(f"\n🎲 Model Ensemble Weights:")
    for model, weight in results['model_weights'].items():
        print(f"  {model.upper()}: {weight:.3f}")
    
    print(f"\n💰 Backtest Results:")
    bt = results['backtest_results']
    bp = bt['betting_performance']
    
    print(f"  Model Accuracy: {bt['accuracy']:.3f}")
    print(f"  Total Bets: {bp.get('total_bets', 0)}")
    print(f"  Win Rate: {bp.get('win_rate', 0):.1f}%")
    print(f"  ROI: {bp.get('roi', 0):.1f}%")
    print(f"  Bankroll Growth: {bp.get('bankroll_growth', 0):.1f}%")
    print(f"  Final Bankroll: £{bp.get('current_bankroll', bankroll):.2f}")
    
    # Feature analysis
    print(f"\n🔍 Top Performing Features:")
    feature_analysis = system.get_feature_analysis()
    
    if 'top_features' in feature_analysis:
        # Show top 5 features from Random Forest (usually most interpretable)
        rf_features = feature_analysis['top_features'].get('rf', [])[:5]
        for i, (feature, importance) in enumerate(rf_features, 1):
            print(f"  {i}. {feature}: {importance:.4f}")
    
    # Save system
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    system.save_system(f'betting_system_{timestamp}.pkl')
    
    print(f"\n✅ System trained and saved successfully!")
    print(f"📈 Expected Performance: {bt['accuracy']:.1%} accuracy, {bp.get('roi', 0):.1f}% ROI")
    
    return system

class PerformanceDashboard:
    """
    Real-time performance tracking dashboard
    """
    
    def __init__(self, system: ProductionBettingSystem):
        self.system = system
        self.live_tracking = {
            'daily_performance': [],
            'weekly_performance': [],
            'monthly_performance': []
        }
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        
        kelly_stats = self.system.kelly_betting.get_performance_stats()
        
        if not kelly_stats:
            return "No betting history available."
        
        report = f"""
🎯 BETTING SYSTEM PERFORMANCE REPORT
{'=' * 50}

💰 FINANCIAL PERFORMANCE
Current Bankroll: £{kelly_stats['current_bankroll']:.2f}
Initial Bankroll: £{self.system.kelly_betting.initial_bankroll:.2f}
Total Profit/Loss: £{kelly_stats['total_profit']:.2f}
Bankroll Growth: {kelly_stats['bankroll_growth']:.1f}%

📊 BETTING STATISTICS  
Total Bets Placed: {kelly_stats['total_bets']}
Winning Bets: {kelly_stats['wins']}
Losing Bets: {kelly_stats['losses']}
Win Rate: {kelly_stats['win_rate']:.1f}%

💹 EFFICIENCY METRICS
Total Staked: £{kelly_stats['total_staked']:.2f}
Return on Investment: {kelly_stats['roi']:.1f}%
Average Bet Size: £{kelly_stats['total_staked']/kelly_stats['total_bets']:.2f}

🎲 RISK MANAGEMENT
Kelly Criterion: ✅ Active
Max Bet Fraction: {self.system.kelly_betting.max_bet_fraction:.1%}
Bankroll Protection: ✅ Implemented

📈 MODEL PERFORMANCE
Ensemble Accuracy: Available after training
Feature Count: {len(self.system.ensemble.feature_names) if self.system.ensemble else 'N/A'}
Model Confidence: Dynamic threshold active
"""
        return report
    
    def export_results_csv(self, filename: str = None):
        """Export betting results to CSV for analysis"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'betting_results_{timestamp}.csv'
        
        bet_history = self.system.kelly_betting.bet_history
        
        if not bet_history:
            print("No betting history to export.")
            return
        
        # Convert to DataFrame and save
        df = pd.DataFrame(bet_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.to_csv(filename, index=False)
        
        print(f"Results exported to {filename}")

class LiveBettingInterface:
    """
    Interface for live betting recommendations
    """
    
    def __init__(self, system: ProductionBettingSystem):
        self.system = system
    
    def analyze_upcoming_matches(self, match_data: Dict) -> str:
        """Analyze upcoming matches and provide recommendations"""
        
        try:
            recommendations = self.system.get_recommendations(match_data)
            
            if not recommendations:
                return "No profitable betting opportunities found."
            
            report = "🎯 BETTING RECOMMENDATIONS\n"
            report += "=" * 40 + "\n\n"
            
            for i, match_rec in enumerate(recommendations, 1):
                report += f"MATCH {i}:\n"
                
                for rec in match_rec['recommendations']:
                    outcome_names = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
                    
                    report += f"  💡 {outcome_names[rec['outcome']]}:\n"
                    report += f"     Model Probability: {rec['model_probability']:.1%}\n"
                    report += f"     Market Odds: {rec['market_odds']:.2f}\n"
                    report += f"     Value Edge: {rec['edge']:.1%}\n"
                    report += f"     Confidence: {rec['confidence']:.3f}\n"
                    report += f"     Recommended Bet: £{rec['recommended_bet']:.2f}\n"
                    report += f"     Kelly %: {rec['kelly_fraction']:.1%}\n\n"
            
            return report
            
        except Exception as e:
            return f"Error analyzing matches: {str(e)}"
    
    def quick_match_analysis(self, home_team: str, away_team: str, 
                           odds_home: float, odds_draw: float, odds_away: float) -> str:
        """Quick analysis for a single match"""
        
        # This would need match feature data - simplified for demo
        analysis = f"""
🔍 QUICK MATCH ANALYSIS
{home_team} vs {away_team}

Market Odds:
Home: {odds_home:.2f} ({1/odds_home:.1%})
Draw: {odds_draw:.2f} ({1/odds_draw:.1%}) 
Away: {odds_away:.2f} ({1/odds_away:.1%})

Market Margin: {(1/odds_home + 1/odds_draw + 1/odds_away - 1):.1%}

⚠️  Note: Full analysis requires complete match data
Use analyze_upcoming_matches() for complete recommendations.
"""
        
        return analysis

# Usage Examples and Integration Functions

def demo_system_with_sample_data():
    """Demonstrate the system with sample data"""
    
    print("🚀 PREMIER LEAGUE BETTING SYSTEM DEMO")
    print("=" * 50)
    
    try:
        # Run the complete system
        system = run_complete_system('data/combined_all_seasons.csv', 1000)
        
        # Create dashboard
        dashboard = PerformanceDashboard(system)
        
        # Show performance report
        print(dashboard.generate_performance_report())
        
        # Export results
        dashboard.export_results_csv()
        
        # Demo live interface
        live_interface = LiveBettingInterface(system)
        
        print("\n" + "=" * 50)
        print("📱 LIVE BETTING INTERFACE READY")
        print("=" * 50)
        print("System is ready for live match analysis!")
        print("Use live_interface.analyze_upcoming_matches(match_data) for recommendations")
        
        return system, dashboard, live_interface
        
    except FileNotFoundError:
        print("❌ Error: data/combined_all_seasons.csv not found")
        print("Please ensure your Premier League data file is available")
        return None, None, None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None, None, None

def create_sample_match_data():
    """Create sample match data for testing"""
    
    # Sample data structure that matches our feature requirements
    sample_data = {
        'odds-home': np.array([2.1, 1.8, 3.2]),
        'odds-draw': np.array([3.4, 3.6, 3.1]),
        'odds-away': np.array([3.8, 4.1, 2.3]),
        
        # Basic features (you would need all features from your trained model)
        'home-ppg-5': np.array([2.2, 1.8, 1.4]),
        'away-ppg-5': np.array([1.6, 2.0, 2.4]),
        'home-gd-5': np.array([0.8, 0.2, -0.4]),
        'away-gd-5': np.array([-0.2, 0.6, 1.2]),
        'big6-home': np.array([1, 1, 0]),
        'big6-away': np.array([0, 1, 1]),
        'home-rest-days': np.array([3, 4, 3]),
        'away-rest-days': np.array([3, 3, 4]),
    }
    
    return sample_data

def main():
    """Main execution function"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🎯 PREMIER LEAGUE BETTING SYSTEM")
    print("Professional-Grade Prediction & Betting Platform")
    print("=" * 60)
    
    # Demo the complete system
    system, dashboard, live_interface = demo_system_with_sample_data()
    
    if system:
        print("\n✅ SYSTEM READY FOR PRODUCTION")
        print("\nNext Steps:")
        print("1. Update data/combined_all_seasons.csv with latest match data")
        print("2. Run system.train_system() to retrain with new data")
        print("3. Use live_interface for upcoming match analysis")
        print("4. Monitor performance with dashboard.generate_performance_report()")
        
        # Example of how to use with new match data
        print("\n🔍 SAMPLE RECOMMENDATION:")
        sample_data = create_sample_match_data()
        recommendations = live_interface.analyze_upcoming_matches(sample_data)
        print(recommendations)

if __name__ == "__main__":
    main()