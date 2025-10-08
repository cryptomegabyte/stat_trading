# 🚀 Crypto Trading Bot with ML Predictions

## ⚖️ Disclaimer

This is experimental software. Trading cryptocurrencies involves significant risk. Use at your own risk. The authors are not responsible for any financial losses.

A sophisticated crypto trading bot that uses machine learning to predict price movements and execute trades on **Kraken**. Built with Rust and barter-rs ecosystem.

## ✨ Features

- **Advanced ML Models**: GAS (Generalized Autoregressive Score) models with heavy-tailed distributions (GHD, VG, NIG, GLD)
- **Bayesian Analysis**: Model comparison and uncertainty quantification using Bayesian statistics
- **Hybrid EGARCH-LSTM**: Combined volatility modeling and deep learning for superior predictions
- **Real-time Trading**: Connects to Kraken WebSocket for real-time BTC/USD data
- **Multi-Pair Support**: Trades BTC, ETH, XRP, SOL, BNB, LTC simultaneously
- **Risk Management**: Dynamic position sizing, stop-loss, take-profit with Bayesian risk assessment
- **Backtesting & Paper Trading**: Comprehensive testing modes with realistic market simulation

## 📊 Performance

## 🏗️ Architecture

- **ML Models**: GAS variants (GHD, VG, NIG, GLD) with heavy-tailed distributions
- **Bayesian Framework**: Model comparison using Bayes factors and credible intervals
- **Hybrid System**: EGARCH volatility modeling combined with LSTM neural networks
- **Features**: Price momentum, volume changes, volatility measures, technical indicators
- **Prediction Horizon**: Multi-step ahead price movement predictions
- **Data Source**: Real-time Kraken API with 1-year historical data for backtesting

## 🚀 Quick Start

### Backtesting (Safe)

```bash
cargo run -- --backtest
```

### Paper Trading (Safe Simulation)

```bash
cargo run -- --paper-trading
```

### Live Trading (Real Money!)

```bash
# Set Kraken API credentials (optional - runs in simulation mode without them)
export KRAKEN_API_KEY="your_api_key"
export KRAKEN_SECRET_KEY="your_secret_key"

# Run live trading
cargo run
```

⚠️ **WARNING**: Live trading will execute real orders on Kraken. Start with small amounts!

## 🛠️ Development

### Prerequisites

- Rust 1.70+
- Binance account with API keys (for live trading)

### Build

```bash
cargo build --release
```

### Test

```bash
cargo test
```

## 📈 Trading Strategy

1. **Data Collection**: Streams real-time trades from Kraken WebSocket API
2. **Feature Engineering**: Calculates momentum, volatility, and technical indicators
3. **Model Training**: Trains multiple ML models (GAS variants, Random Forest, Hybrid EGARCH-LSTM)
4. **Bayesian Model Selection**: Uses Bayes factors to select best performing model
5. **Signal Generation**: Generates BUY/SELL/HOLD signals based on ensemble predictions
6. **Risk Management**: Dynamic position sizing with stop-loss and take-profit
7. **Portfolio Optimization**: Allocates capital across multiple trading pairs

## 🔧 Configuration

- **Trading Pairs**: BTC/USD, SOL/USD, XRP/USD, LTC/USD (configurable)
- **Starting Balance**: $500 per pair ($2000 total for paper/live trading)
- **Stop Loss**: 1.5% (conservative risk management)
- **Take Profit**: 4.0% (allows profits to run)
- **Max Position Size**: 10% of pair balance
- **Training Threshold**: 50 trades minimum before ML model activation
- **Model Selection**: Automatic Bayesian model comparison (gas_vg default)

## 📊 Monitoring

The bot provides real-time logging:

- Trade executions
- P&L updates
- Risk management triggers
- Model predictions
- Account balance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request


