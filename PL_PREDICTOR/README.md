# 🏈 Premier League Prediction System

A comprehensive machine learning system for predicting Premier League match outcomes with live data integration and betting recommendations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Integration](#api-integration)
- [Model Performance](#model-performance)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This system combines historical Premier League data analysis with live API integration to provide accurate match predictions and betting recommendations. It features:

- **180+ Advanced Features**: Psychological, statistical, and momentum-based features
- **Ensemble ML Models**: Random Forest, Gradient Boosting, and Neural Networks
- **Live Data Integration**: Real-time fixture data from football-data.org API
- **Betting Recommendations**: Confidence-based betting suggestions with ROI tracking

## ✨ Features

### 🧠 Advanced Analytics
- **180+ Feature Engineering**: Team form, head-to-head stats, rest days, momentum
- **Psychological Factors**: Big 6 team analysis, home/away performance
- **Statistical Modeling**: 17 seasons of historical data analysis
- **Ensemble Learning**: Multiple ML models for improved accuracy

### 📊 Live Predictions
- **Real-time Fixtures**: Live Premier League match data
- **Win/Draw/Loss Probabilities**: Detailed outcome predictions
- **Confidence Scoring**: Betting confidence levels (Strong/Moderate/Weak)
- **Team Strength Analysis**: Dynamic team performance ratings

### 💰 Betting System
- **ROI Tracking**: Historical performance monitoring
- **Bankroll Management**: Risk-adjusted betting recommendations
- **Value Betting**: Identification of profitable betting opportunities
- **Performance Analytics**: Detailed betting results and statistics

## 🏗️ System Architecture

```
PL_PREDICTOR/
├── 📊 Data Processing
│   ├── enhanced_dataset.py      # Historical data processing
│   ├── advanced_features.py     # 180+ feature engineering
│   └── data_merger.py          # Data consolidation
├── 🤖 Machine Learning
│   ├── production_betting_system.py  # Ensemble ML models
│   └── enhanced_predict.py     # Prediction pipeline
├── 🌐 Live Integration
│   ├── live_data_fetcher.py    # API integration
│   └── live_betting_runner.py  # Main orchestrator
└── 🚀 Quick Access
    └── get_predictions.py      # Simple prediction script
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Internet connection for API access

### Setup Instructions

1. **Clone/Download the Repository**
   ```bash
   cd PL_PREDICTOR
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set API Key** (Optional - for live data)
   ```bash
   # Set environment variable
   export FOOTBALL_DATA_API_KEY="your_api_key_here"
   
   # Or edit the script directly
   # API key is already configured in the scripts
   ```

## ⚡ Quick Start

### Get Immediate Predictions
```bash
python get_predictions.py
```

**Output Example:**
```
🏈 MATCH 1: Liverpool vs Bournemouth
📅 Date: 2025-08-15 19:00 UTC
🎯 PREDICTION: Liverpool Win (44.3% confidence)
📊 PROBABILITIES:
   • Home Win (Liverpool): 44.3%
   • Draw: 37.3%
   • Away Win (Bournemouth): 18.4%
💰 BETTING SUGGESTION: 💛 MODERATE BET: Liverpool Win
```

### Run Full System Analysis
```bash
python live_betting_runner.py
```

## 📖 Usage

### 1. Simple Predictions (`get_predictions.py`)
- **Purpose**: Quick match predictions without complex analysis
- **Best for**: Getting immediate betting insights
- **Features**: Team strength analysis, basic probabilities

### 2. Full System Analysis (`live_betting_runner.py`)
- **Purpose**: Complete analysis with historical data integration
- **Best for**: Detailed betting analysis and system training
- **Features**: 180+ features, ensemble models, ROI tracking

### 3. Custom Analysis
```python
from live_data_fetcher import FootballDataFetcher, LiveBettingIntegration

# Initialize system
fetcher = FootballDataFetcher("your_api_key")
integration = LiveBettingIntegration(
    api_key="your_api_key",
    historical_data_path='data/combined_all_seasons.csv'
)

# Get predictions
predictions = integration.get_live_recommendations(days_ahead=7)
print(predictions)
```

## 🌐 API Integration

### Football-Data.org API
- **Rate Limit**: 10 requests/minute
- **Data**: Live Premier League fixtures, team stats
- **Authentication**: API key required
- **Caching**: 5-minute response caching

### API Features
- ✅ Upcoming match fixtures
- ✅ Team statistics and form
- ✅ Match dates and times
- ✅ Real-time data updates

## 📈 Model Performance

### Current Performance Metrics
- **Model Accuracy**: ~24.5%
- **Backtest ROI**: -21.5%
- **Feature Count**: 180+ engineered features
- **Training Data**: 17 seasons of Premier League data

### Model Components
1. **Random Forest**: Base predictions
2. **Gradient Boosting**: Advanced pattern recognition
3. **Neural Network**: Complex feature relationships

## 📁 File Structure

```
PL_PREDICTOR/
├── 📊 Core System
│   ├── enhanced_dataset.py          # Historical data processing
│   ├── advanced_features.py         # Feature engineering (180+ features)
│   ├── production_betting_system.py # ML ensemble models
│   └── enhanced_predict.py          # Prediction pipeline
├── 🌐 Live Integration
│   ├── live_data_fetcher.py         # API integration & data fetching
│   └── live_betting_runner.py       # Main system orchestrator
├── 🚀 Quick Access
│   └── get_predictions.py           # Simple prediction script
├── 📁 Data
│   └── combined_all_seasons.csv     # Historical match data
├── 📋 Configuration
│   ├── requirements.txt             # Python dependencies
│   └── README.md                    # This file
└── 📊 Output
    ├── betting_system.log           # System logs
    ├── betting_results_*.csv        # Betting performance
    └── betting_system_*.pkl         # Trained models
```

## ⚙️ Configuration

### Environment Variables
```bash
FOOTBALL_DATA_API_KEY=your_api_key_here
```

### API Configuration
- **Base URL**: `https://api.football-data.org/v4`
- **Rate Limit**: 10 requests/minute
- **Cache Duration**: 5 minutes
- **Default League**: Premier League (2021)

### Model Parameters
- **Training Split**: 80% train, 20% test
- **Cross Validation**: 5-fold
- **Feature Selection**: Top 180+ features
- **Ensemble Weights**: Optimized for accuracy

## 🔧 Troubleshooting

### Common Issues

1. **API Connection Errors**
   ```bash
   # Check API key
   echo $FOOTBALL_DATA_API_KEY
   
   # Test connection
   python -c "from live_data_fetcher import test_api_connection; test_api_connection('your_key')"
   ```

2. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

3. **Data Loading Issues**
   ```bash
   # Check data file exists
   ls -la data/combined_all_seasons.csv
   
   # Verify file integrity
   python -c "import pandas as pd; df = pd.read_csv('data/combined_all_seasons.csv'); print(f'Loaded {len(df)} rows')"
   ```

4. **Timezone Errors**
   - The system handles timezone conversion automatically
   - All dates are converted to UTC for consistency

### Performance Optimization
- **Memory Usage**: ~2GB RAM recommended
- **Processing Time**: ~30 seconds for full analysis
- **API Calls**: Minimized through caching

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings for all functions
- Include error handling
- Write unit tests for new features

## 📄 License

This project is for educational and research purposes. Please ensure compliance with:
- Football-Data.org API terms of service
- Data usage policies

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review system logs in `betting_system.log`
3. Verify API key and connectivity
4. Test with simple prediction script first

---

**⚡ Ready to predict Premier League matches? Run `python get_predictions.py` to get started!**



*Built in Coimbatore 🇮🇳 | by Makilesh M*