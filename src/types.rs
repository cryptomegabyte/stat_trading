#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TradingPair {
    BTC,
    ETH,
    XRP,
    SOL,
    BNB,
    LTC,
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
        }
    }

    pub fn kraken_pair(&self) -> &'static str {
        match self {
            TradingPair::BTC => "XBTUSD",
            TradingPair::ETH => "ETHUSD",
            TradingPair::XRP => "XRPUSD",
            TradingPair::SOL => "SOLUSD",
            TradingPair::BNB => "BNBUSD",
            TradingPair::LTC => "LTCUSD",
        }
    }
}

#[derive(Debug)]
pub struct TradingConfig {
    pub pairs: Vec<TradingPair>,
    pub initial_balance: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub max_position_size_pct: f64,
    pub leverage: f64,
}

impl Default for TradingConfig {
    fn default() -> Self {
           // Focus on top performing pairs: BTC, SOL, XRP, LTC for live trading
           let pairs = vec![TradingPair::BTC, TradingPair::SOL, TradingPair::XRP, TradingPair::LTC];
        let total_balance = 2000.0;
        let initial_balance = total_balance / pairs.len() as f64;

        Self {
            pairs,
            initial_balance,
            stop_loss_pct: 1.5,         // Conservative: 1.5% stop-loss
            take_profit_pct: 4.0,       // Conservative: 4% take profit
            max_position_size_pct: 0.1, // Conservative: 10% max position size
            leverage: 1.0,              // SPOT TRADING: No leverage (1.0 = no amplification)
        }
    }
}

#[derive(Debug, Clone)]
pub struct TradeData {
    pub price: f64,
    pub volume: f64,
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
