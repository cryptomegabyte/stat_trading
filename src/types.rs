#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TradingPair {
    BTC,
    ETH,
    XRP,
    SOL,
    BNB,
    LTC,
    ADA,
    DOT,
    LINK,
    AVAX,
    MATIC,
    DOGE,
    SHIB,
    UNI,
    AAVE,
}

impl TradingPair {
    pub fn symbol(&self) -> &'static str {
        match self {
            TradingPair::BTC => "btc",
            TradingPair::ETH => "eth",
            TradingPair::XRP => "xrp",
            TradingPair::SOL => "sol",
            TradingPair::BNB => "bnb",
            TradingPair::LTC => "ltc",
            TradingPair::ADA => "ada",
            TradingPair::DOT => "dot",
            TradingPair::LINK => "link",
            TradingPair::AVAX => "avax",
            TradingPair::MATIC => "matic",
            TradingPair::DOGE => "doge",
            TradingPair::SHIB => "shib",
            TradingPair::UNI => "uni",
            TradingPair::AAVE => "aave",
        }
    }

    pub fn quote_symbol(&self) -> &'static str {
        "usdt"
    }

    pub fn kraken_symbol(&self) -> &'static str {
        match self {
            TradingPair::BTC => "XXBTZUSD",
            TradingPair::ETH => "XETHZUSD",
            TradingPair::XRP => "XXRPZUSD",
            TradingPair::SOL => "SOLUSD",
            TradingPair::BNB => "BNBUSD",
            TradingPair::LTC => "XLTCZUSD",
            TradingPair::ADA => "ADAUSD",
            TradingPair::DOT => "DOTUSD",
            TradingPair::LINK => "LINKUSD",
            TradingPair::AVAX => "AVAXUSD",
            TradingPair::MATIC => "MATICUSD",
            TradingPair::DOGE => "DOGEUSD",
            TradingPair::SHIB => "SHIBUSD",
            TradingPair::UNI => "UNIUSD",
            TradingPair::AAVE => "AAVEUSD",
        }
    }

    pub fn kraken_pair(&self) -> &'static str {
        match self {
            TradingPair::BTC => "XBT/USD",
            TradingPair::ETH => "ETH/USD",
            TradingPair::XRP => "XRP/USD",
            TradingPair::SOL => "SOL/USD",
            TradingPair::BNB => "BNB/USD",
            TradingPair::LTC => "LTC/USD",
            TradingPair::ADA => "ADA/USD",
            TradingPair::DOT => "DOT/USD",
            TradingPair::LINK => "LINK/USD",
            TradingPair::AVAX => "AVAX/USD",
            TradingPair::MATIC => "MATIC/USD",
            TradingPair::DOGE => "DOGE/USD",
            TradingPair::SHIB => "SHIB/USD",
            TradingPair::UNI => "UNI/USD",
            TradingPair::AAVE => "AAVE/USD",
        }
    }
}

#[derive(Debug)]
pub struct TradingConfig {
    pub pairs: Vec<TradingPair>,
    pub initial_balances: Vec<f64>, // Individual balance per pair
    pub total_balance: f64, // Total across all pairs
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub max_position_size_pct: f64,
    pub leverage: f64,
    pub max_drawdown_pct: f64,
    pub max_daily_trades: u32,
    pub max_consecutive_losses: u32,
    pub volatility_adjustment: bool,
    pub correlation_limits: std::collections::HashMap<String, f64>,
}

impl Default for TradingConfig {
    fn default() -> Self {
        // Risk-based capital allocation: more capital to historically better performing pairs
        let pairs = vec![
            TradingPair::BTC,   // 20% allocation - most liquid, best performance
            TradingPair::ETH,   // 15% allocation - strong fundamentals
            TradingPair::SOL,   // 12% allocation - high growth potential
            TradingPair::BNB,   // 8% allocation - ecosystem strength
            TradingPair::XRP,   // 8% allocation - proven performer
            TradingPair::ADA,   // 6% allocation - smart contract platform
            TradingPair::DOT,   // 5% allocation - interoperability focus
            TradingPair::LINK,  // 4% allocation - oracle network
            TradingPair::AVAX,  // 3% allocation - high throughput
            TradingPair::MATIC, // 3% allocation - layer 2 scaling
            TradingPair::DOGE,  // 3% allocation - meme coin with real adoption
            TradingPair::UNI,   // 3% allocation - DEX leader
            TradingPair::AAVE,  // 3% allocation - DeFi lending
            TradingPair::SHIB,  // 3% allocation - community driven
            TradingPair::LTC,   // 4% allocation - conservative allocation
        ];
        let total_balance = 2000.0;

        // Risk-based allocation weights (sum to 1.0)
        let allocation_weights = [
            0.20, // BTC
            0.15, // ETH
            0.12, // SOL
            0.08, // BNB
            0.08, // XRP
            0.06, // ADA
            0.05, // DOT
            0.04, // LINK
            0.03, // AVAX
            0.03, // MATIC
            0.03, // DOGE
            0.03, // UNI
            0.03, // AAVE
            0.03, // SHIB
            0.04, // LTC
        ];

        // Calculate initial balance per pair based on risk allocation
        let mut initial_balances = Vec::new();
        for &weight in &allocation_weights {
            initial_balances.push(total_balance * weight);
        }

        let mut correlation_limits = std::collections::HashMap::new();
        correlation_limits.insert("BTC_ETH".to_string(), 0.8);
        correlation_limits.insert("BTC_SOL".to_string(), 0.6);
        correlation_limits.insert("ETH_SOL".to_string(), 0.7);

        Self {
            pairs,
            initial_balances,
            total_balance,
            stop_loss_pct: 1.5,         // Conservative: 1.5% stop-loss
            take_profit_pct: 4.0,       // Conservative: 4% take profit
            max_position_size_pct: 0.25, // Increased from 15% to 25% for meaningful monthly returns
            leverage: 2.0,              // Increased to 2.0x leverage for higher profit potential
            max_drawdown_pct: 5.0,      // Max 5% drawdown before halting
            max_daily_trades: 20,       // Max 20 trades per day
            max_consecutive_losses: 3,  // Max 3 consecutive losses
            volatility_adjustment: true, // Adjust position sizes based on volatility
            correlation_limits,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TradeData {
    pub price: f64,
    pub volume: f64,
}

#[derive(Debug, Clone)]
pub struct MultiTimeframeData {
    pub timeframe_15m: Vec<(f64, f64)>, // (price, volume)
    pub timeframe_1h: Vec<(f64, f64)>,  // (price, volume)
    pub timeframe_4h: Vec<(f64, f64)>,  // (price, volume)
}

#[derive(Debug)]
pub enum MLModel {
    LinearRegression(linfa_linear::FittedLinearRegression<f64>),
    RandomForest(
        smartcore::ensemble::random_forest_regressor::RandomForestRegressor<
            f64,
            f64,
            smartcore::linalg::basic::matrix::DenseMatrix<f64>,
            Vec<f64>,
        >,
    ),
    HybridEGARCHLSTM(Box<crate::ml::HybridEGARCHLSTM>),
    GAS(Box<crate::ml::GASModel>),
    HybridGASRF(Box<crate::ml::HybridGASRF>),
    Ensemble(Box<crate::ml::BayesianEnsemblePredictor>),
}
