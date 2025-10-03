# 🚀 Crypto Trading Bot with ML Predictions

A sophisticated crypto trading bot that uses machine learning to predict price movements and execute trades on Binance. Built with Rust and barter-rs ecosystem.

## ✨ Features

- **Real-time ML Predictions**: Linear regression model trained on momentum-based features
- **Live Trading**: Connects to Binance WebSocket for real-time BTC/USDT data
- **Risk Management**: 1% stop-loss, 2% take-profit, 10% position sizing
- **Backtesting**: Comprehensive backtesting with simulated market data
- **Trading Signals**: BUY/SELL/HOLD signals based on price predictions

## 📊 Performance

**Backtest Results:**
- 17 trades executed
- 35.29% win rate
- Risk-managed position sizing
- Ready for live deployment

## 🏗️ Architecture

- **ML Model**: Linear regression with z-score normalization
- **Features**: Price momentum, volume changes, SMA momentum
- **Prediction Horizon**: 3-trade ahead price change prediction
- **Data Source**: Real-time Binance BTC/USDT spot trades

## 🚀 Quick Start

### Backtesting (Safe)

```bash
cargo run -- --backtest
```

### Live Trading (Real Money!)

```bash
# Set API credentials (optional - runs in simulation mode without them)
export BINANCE_API_KEY="your_api_key"
export BINANCE_SECRET_KEY="your_secret_key"

# Run live trading
cargo run
```

⚠️ **WARNING**: Live trading will execute real orders on Binance. Start with small amounts!

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

1. **Data Collection**: Streams real-time trades from Binance
2. **Feature Engineering**: Calculates momentum indicators
3. **ML Training**: Trains on historical price patterns
4. **Signal Generation**: Predicts price changes and generates trading signals
5. **Risk Management**: Automatic stop-loss and take-profit execution
6. **Position Sizing**: Limits exposure to 10% of balance per trade

## 🔧 Configuration

- **Starting Balance**: $1000 (live), $10000 (backtest)
- **Stop Loss**: 1%
- **Take Profit**: 2%
- **Max Position Size**: 10% of balance
- **Training Threshold**: 50 trades minimum

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

## ⚖️ Disclaimer

This is experimental software. Trading cryptocurrencies involves significant risk. Use at your own risk. The authors are not responsible for any financial losses.

## 📄 License

MIT License - see LICENSE file for details.
