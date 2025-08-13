import tensorflow as tf
import numpy as np
import csv
import logging
from enhanced_dataset import EnhancedDataset
# import betting

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.basicConfig(level=logging.INFO)

TRAINING_SET_FRACTION = 0.8

def create_feature_columns(feature_names):
    """Create TensorFlow feature columns from feature names"""
    feature_columns = []
    
    # Numeric features - all our enhanced features are numeric
    for feature_name in feature_names:
        if 'odds' not in feature_name:  # Exclude odds from features (they're targets)
            feature_columns.append(
                tf.feature_column.numeric_column(key=feature_name)
            )
    
    return feature_columns

def create_input_functions(train_features, train_labels, test_features, test_labels, batch_size=500):
    """Create TensorFlow input functions with proper data handling"""
    
    # Filter out odds columns from features (keep them in test_features for betting)
    feature_keys = [key for key in train_features.keys() if 'odds' not in key]
    
    train_features_filtered = {key: train_features[key] for key in feature_keys}
    test_features_filtered = {key: test_features[key] for key in feature_keys}
    
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x=train_features_filtered,
        y=train_labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )
    
    test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x=test_features_filtered,
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )
    
    return train_input_fn, test_input_fn

def create_enhanced_model(feature_columns, model_dir='enhanced_model/'):
    """Create an enhanced neural network model with better architecture"""
    
    # Enhanced architecture - deeper network for complex patterns
    model = tf.estimator.DNNClassifier(
        model_dir=model_dir,
        hidden_units=[64, 32, 16],  # Deeper network for complex relationships
        feature_columns=feature_columns,
        n_classes=3,
        label_vocabulary=['H', 'D', 'A'],
        optimizer=tf.train.AdamOptimizer(  # Adam is generally better than ProximalAdagrad
            learning_rate=0.001
        ),
        dropout=0.2,  # Add dropout for regularization
        activation_fn=tf.nn.relu
    )
    
    return model

def enhanced_betting_strategy(predictions, test_features, test_labels, 
                            edge_thresholds={'H': 0.05, 'D': 0.05, 'A': 0.05},
                            confidence_threshold=0.6):
    """
    Enhanced betting strategy with confidence-based thresholds
    and separate edge thresholds for each outcome
    """
    result = {
        'total_bets': 0,
        'home_bets': {'count': 0, 'wins': 0, 'stake': 0, 'return': 0},
        'draw_bets': {'count': 0, 'wins': 0, 'stake': 0, 'return': 0},
        'away_bets': {'count': 0, 'wins': 0, 'stake': 0, 'return': 0},
        'total_stake': 0,
        'total_return': 0
    }
    
    for i, prediction in enumerate(predictions):
        probabilities = prediction['probabilities']
        max_prob_idx = np.argmax(probabilities)
        max_probability = probabilities[max_prob_idx]
        
        # Only bet if we have sufficient confidence
        if max_probability < confidence_threshold:
            continue
        
        # Check each outcome for betting opportunities
        outcomes = ['H', 'D', 'A']
        odds_keys = ['odds-home', 'odds-draw', 'odds-away']
        bet_categories = ['home_bets', 'draw_bets', 'away_bets']
        
        for j, (outcome, odds_key, bet_category) in enumerate(zip(outcomes, odds_keys, bet_categories)):
            model_prob = probabilities[j]
            market_odds = test_features[odds_key][i]
            
            if market_odds <= 1:  # Skip invalid odds
                continue
                
            implied_prob = 1 / market_odds
            edge = model_prob - implied_prob
            
            # Bet if we have sufficient edge
            if edge > edge_thresholds[outcome]:
                stake = 1  # Unit stake
                result['total_bets'] += 1
                result[bet_category]['count'] += 1
                result[bet_category]['stake'] += stake
                result['total_stake'] += stake
                
                # Check if bet won
                if test_labels[i] == outcome:
                    payout = stake * market_odds
                    result[bet_category]['wins'] += 1
                    result[bet_category]['return'] += payout
                    result['total_return'] += payout
    
    # Calculate performance metrics
    if result['total_stake'] > 0:
        result['roi'] = (result['total_return'] - result['total_stake']) / result['total_stake'] * 100
        result['profit'] = result['total_return'] - result['total_stake']
        result['win_rate'] = (result['home_bets']['wins'] + result['draw_bets']['wins'] + 
                             result['away_bets']['wins']) / result['total_bets'] * 100
    else:
        result['roi'] = 0
        result['profit'] = 0
        result['win_rate'] = 0
    
    return result

def main():
    """Enhanced main function with comprehensive training and evaluation"""
    
    print("🚀 Loading Enhanced Dataset...")
    # Load enhanced dataset
    dataset = EnhancedDataset('data/combined_all_seasons.csv')
    
    # Print dataset summary
    summary = dataset.get_data_summary()
    print(f"\n📊 Dataset Summary:")
    print(f"Total matches: {summary['total_matches']}")
    print(f"Features: {summary['feature_count']}")
    print(f"Date range: {summary['date_range'][0].strftime('%Y-%m-%d')} to {summary['date_range'][1].strftime('%Y-%m-%d')}")
    print(f"Draw rate: {summary['draw_pct']:.1f}%")
    
    # Get training data
    train_features, train_labels, test_features, test_labels = dataset.get_training_data(
        train_fraction=TRAINING_SET_FRACTION
    )
    
    print(f"\n🎯 Training samples: {len(train_labels)}")
    print(f"Testing samples: {len(test_labels)}")
    
    # Get feature names (excluding odds)
    all_features = dataset.get_feature_names()
    model_features = [f for f in all_features if 'odds' not in f]
    
    print(f"\n🔧 Model features: {len(model_features)}")
    print("New enhanced features:")
    enhanced_features = [f for f in model_features if any(keyword in f for keyword in 
                        ['rest', 'momentum', 'big-6', 'h2h', '-3', '-5', '-10'])]
    for feature in enhanced_features[:10]:  # Show first 10
        print(f"  - {feature}")
    if len(enhanced_features) > 10:
        print(f"  ... and {len(enhanced_features) - 10} more")
    
    # Create feature columns and input functions
    feature_columns = create_feature_columns(model_features)
    train_input_fn, test_input_fn = create_input_functions(
        train_features, train_labels, test_features, test_labels
    )
    
    # Create enhanced model
    print(f"\n🧠 Building Enhanced Neural Network...")
    model = create_enhanced_model(feature_columns)
    
    # Training loop with detailed logging
    print(f"\n🏋️ Starting Training...")
    
    best_accuracy = 0
    best_roi = -100
    training_log = []
    
    with open('enhanced_training_log.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Step', 'Accuracy', 'Loss', 'ROI', 'Profit', 'Win_Rate', 'Total_Bets'])
        
        for i in range(50):  # Reduced iterations for faster testing
            # Train model
            model.train(input_fn=train_input_fn, steps=200)
            
            # Evaluate model
            evaluation_result = model.evaluate(input_fn=test_input_fn)
            accuracy = evaluation_result['accuracy']
            loss = evaluation_result['average_loss']
            
            # Get predictions for betting analysis
            predictions = list(model.predict(input_fn=test_input_fn))
            
            # Test betting strategy
            betting_result = enhanced_betting_strategy(
                predictions, test_features, test_labels,
                edge_thresholds={'H': 0.03, 'D': 0.05, 'A': 0.03},  # Lower thresholds
                confidence_threshold=0.55  # Lower confidence threshold
            )
            
            # Log results
            step = (i + 1) * 200
            roi = betting_result['roi']
            profit = betting_result['profit']
            win_rate = betting_result['win_rate']
            total_bets = betting_result['total_bets']
            
            writer.writerow([step, accuracy, loss, roi, profit, win_rate, total_bets])
            
            # Track best performance
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            if roi > best_roi:
                best_roi = roi
            
            # Print progress every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"Step {step}: Accuracy={accuracy:.3f}, ROI={roi:.1f}%, "
                     f"Profit={profit:.1f}, Bets={total_bets}")
    
    # Final evaluation
    print(f"\n🎯 Final Results:")
    print(f"Best Accuracy: {best_accuracy:.3f}")
    print(f"Best ROI: {best_roi:.1f}%")
    
    # Detailed betting breakdown
    final_predictions = list(model.predict(input_fn=test_input_fn))
    final_betting = enhanced_betting_strategy(
        final_predictions, test_features, test_labels,
        edge_thresholds={'H': 0.03, 'D': 0.05, 'A': 0.03},
        confidence_threshold=0.55
    )
    
    print(f"\n💰 Betting Performance Breakdown:")
    print(f"Total Bets: {final_betting['total_bets']}")
    print(f"Total Stake: {final_betting['total_stake']:.1f} units")
    print(f"Total Return: {final_betting['total_return']:.1f} units")
    print(f"Profit: {final_betting['profit']:.1f} units")
    print(f"ROI: {final_betting['roi']:.1f}%")
    print(f"Win Rate: {final_betting['win_rate']:.1f}%")
    
    print(f"\nHome Bets: {final_betting['home_bets']['count']} "
          f"(Win Rate: {final_betting['home_bets']['wins']/max(1, final_betting['home_bets']['count'])*100:.1f}%)")
    print(f"Draw Bets: {final_betting['draw_bets']['count']} "
          f"(Win Rate: {final_betting['draw_bets']['wins']/max(1, final_betting['draw_bets']['count'])*100:.1f}%)")
    print(f"Away Bets: {final_betting['away_bets']['count']} "
          f"(Win Rate: {final_betting['away_bets']['wins']/max(1, final_betting['away_bets']['count'])*100:.1f}%)")
    
    return model, final_betting

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()  # For compatibility with estimators
    model, results = main()