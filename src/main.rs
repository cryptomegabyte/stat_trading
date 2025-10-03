use anyhow::Result;
use barter_data::exchange::binance::spot::BinanceSpot;
use barter_data::streams::Streams;
use barter_data::subscription::trade::PublicTrades;
use barter_instrument::instrument::market_data::kind::MarketDataInstrumentKind;
use futures::StreamExt;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use linfa::prelude::*;
use linfa_linear::{LinearRegression, FittedLinearRegression};
use ndarray::{Array1, Array2};
use std::collections::VecDeque;
use tracing::{info, warn, Level};
use tracing_subscriber;
use std::env;

#[derive(Debug, Clone)]
struct TradeData {
    price: f64,
    volume: f64,
}

#[derive(Debug)]
enum MLModel {
    LinearRegression(FittedLinearRegression<f64>),
    RandomForest(RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>),
}

#[derive(Debug)]
struct Backtester {
    predictor: SimpleMLPredictor,
    balance: f64,
    position: f64, // amount of BTC held
    entry_price: Option<f64>,
    total_trades: usize,
    winning_trades: usize,
    total_pnl: f64,
    stop_loss_pct: f64,
    take_profit_pct: f64,
}

impl Backtester {
    fn new() -> Self {
        Self {
            predictor: SimpleMLPredictor::new(50), // increased window size
            balance: 10000.0, // start with $10k
            position: 0.0,
            entry_price: None,
            total_trades: 0,
            winning_trades: 0,
            total_pnl: 0.0,
            stop_loss_pct: 0.01, // 1%
            take_profit_pct: 0.02, // 2%
        }
    }

    fn process_trade(&mut self, price: f64, volume: f64) {
        self.predictor.add_trade(price, volume);
        
        // Check stop-loss and take-profit
        if self.position > 0.0 {
            if let Some(entry) = self.entry_price {
                let pnl_pct = (price - entry) / entry;
                if pnl_pct <= -self.stop_loss_pct {
                    self.sell(price);
                    info!("STOP LOSS triggered at {}%", pnl_pct * 100.0);
                    return;
                } else if pnl_pct >= self.take_profit_pct {
                    self.sell(price);
                    info!("TAKE PROFIT triggered at {}%", pnl_pct * 100.0);
                    return;
                }
            }
        }
        
        if let Some(signal) = self.predictor.get_trading_signal() {
            match signal.as_str() {
                "BUY" => self.buy(price),
                "SELL" => self.sell(price),
                _ => {}
            }
        }
    }

    fn buy(&mut self, price: f64) {
        if self.position == 0.0 && self.balance > 100.0 { // minimum balance
            let trade_amount = self.balance * 0.1; // 10% of balance
            self.position = trade_amount / price;
            self.entry_price = Some(price);
            self.balance -= trade_amount;
            info!("BUY: {} BTC at ${}, position: {}, remaining balance: ${:.2}", self.position, price, self.position, self.balance);
        }
    }

    fn sell(&mut self, price: f64) {
        if self.position > 0.0 {
            let exit_value = self.position * price;
            let pnl = exit_value - (self.entry_price.unwrap() * self.position);
            self.total_pnl += pnl;
            self.balance += exit_value;
            self.total_trades += 1;
            if pnl > 0.0 {
                self.winning_trades += 1;
            }
            info!("SELL: {} BTC at ${}, P&L: ${:.2}, Total P&L: ${:.2}", self.position, price, pnl, self.total_pnl);
            self.position = 0.0;
            self.entry_price = None;
        }
    }

    fn print_results(&self) {
        let win_rate = if self.total_trades > 0 { self.winning_trades as f64 / self.total_trades as f64 * 100.0 } else { 0.0 };
        info!("Backtest Results:");
        info!("Total Trades: {}", self.total_trades);
        info!("Winning Trades: {}", self.winning_trades);
        info!("Win Rate: {:.2}%", win_rate);
        info!("Total P&L: ${:.2}", self.total_pnl);
        info!("Final Balance: ${:.2}", self.balance);
    }
}

#[derive(Debug)]
struct LiveTrader {
    predictor: SimpleMLPredictor,
    balance: f64,
    position: f64, // amount of BTC held
    entry_price: Option<f64>,
    total_trades: usize,
    winning_trades: usize,
    total_pnl: f64,
    stop_loss_pct: f64,
    take_profit_pct: f64,
    max_position_size_pct: f64, // max % of balance to use per trade
}

impl LiveTrader {
    fn new() -> Self {
        Self {
            predictor: SimpleMLPredictor::new(50),
            balance: 1000.0, // start with $1k for safety
            position: 0.0,
            entry_price: None,
            total_trades: 0,
            winning_trades: 0,
            total_pnl: 0.0,
            stop_loss_pct: 0.01, // 1% stop loss
            take_profit_pct: 0.02, // 2% take profit
            max_position_size_pct: 0.1, // 10% of balance max
        }
    }

    fn can_buy(&self, price: f64) -> bool {
        self.position == 0.0 && self.balance >= price * 0.001 // minimum 0.001 BTC
    }

    fn can_sell(&self) -> bool {
        self.position > 0.0
    }

    async fn execute_buy(&mut self, price: f64) -> Result<()> {
        let max_position_value = self.balance * self.max_position_size_pct;
        let btc_to_buy = (max_position_value / price).min(self.balance / price);

        if btc_to_buy * price < 10.0 {
            info!("Trade too small (${:.2}), skipping", btc_to_buy * price);
            return Ok(());
        }

        // Here you would execute the actual buy order
        // For now, we'll simulate it
        info!("🚀 LIVE BUY: {:.6} BTC at ${:.2}, total: ${:.2}",
              btc_to_buy, price, btc_to_buy * price);

        self.position += btc_to_buy;
        self.balance -= btc_to_buy * price;
        self.entry_price = Some(price);
        self.total_trades += 1;

        Ok(())
    }

    async fn execute_sell(&mut self, price: f64) -> Result<()> {
        if self.position <= 0.0 {
            return Ok(());
        }

        let btc_to_sell = self.position;
        let sale_value = btc_to_sell * price;

        // Here you would execute the actual sell order
        // For now, we'll simulate it
        info!("💰 LIVE SELL: {:.6} BTC at ${:.2}, total: ${:.2}",
              btc_to_sell, price, sale_value);

        self.position = 0.0;
        self.balance += sale_value;

        if let Some(entry) = self.entry_price {
            let pnl = sale_value - (btc_to_sell * entry);
            self.total_pnl += pnl;
            if pnl > 0.0 {
                self.winning_trades += 1;
            }
            info!("Trade P&L: ${:.2} ({:.2}%)", pnl, (pnl / (btc_to_sell * entry)) * 100.0);
        }

        self.entry_price = None;

        Ok(())
    }

    async fn process_price_update(&mut self, price: f64) -> Result<()> {
        // Update predictor with new price
        self.predictor.add_trade(price, 1.0); // volume = 1.0 for live updates

        // Check for stop loss / take profit
        if let Some(entry_price) = self.entry_price {
            if self.position > 0.0 {
                let pnl_pct = (price - entry_price) / entry_price;
                if pnl_pct <= -self.stop_loss_pct {
                    info!("🛑 STOP LOSS triggered at {:.2}%", pnl_pct * 100.0);
                    self.execute_sell(price).await?;
                    return Ok(());
                } else if pnl_pct >= self.take_profit_pct {
                    info!("🎯 TAKE PROFIT triggered at {:.2}%", pnl_pct * 100.0);
                    self.execute_sell(price).await?;
                    return Ok(());
                }
            }
        }

        // Get trading signal
        if let Some(signal) = self.predictor.get_trading_signal() {
            match signal.as_str() {
                "BUY" if self.can_buy(price) => {
                    self.execute_buy(price).await?;
                }
                "SELL" if self.can_sell() => {
                    self.execute_sell(price).await?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn print_status(&self) {
        let win_rate = if self.total_trades > 0 {
            self.winning_trades as f64 / self.total_trades as f64 * 100.0
        } else {
            0.0
        };
        info!("📊 Live Trading Status:");
        info!("Balance: ${:.2}", self.balance);
        info!("Position: {:.6} BTC", self.position);
        info!("Total Trades: {}", self.total_trades);
        info!("Win Rate: {:.1}%", win_rate);
        info!("Total P&L: ${:.2}", self.total_pnl);
        if let Some(entry) = self.entry_price {
            info!("Entry Price: ${:.2}", entry);
        }
    }
}

#[derive(Debug)]
struct SimpleMLPredictor {
    model: Option<MLModel>,
    trades: VecDeque<TradeData>,
    window_size: usize,
    feature_means: Vec<f64>,
    feature_stds: Vec<f64>,
    // Rolling statistics for efficiency
    sma5_sum: f64,
    sma10_sum: f64,
    rsi_gains: VecDeque<f64>,
    rsi_losses: VecDeque<f64>,
    ema12: Option<f64>,
    ema26: Option<f64>,
    volatility_sum_sq: f64,
    volume_sma_sum: f64,
}

impl SimpleMLPredictor {
    fn new(window_size: usize) -> Self {
        Self {
            model: None,
            trades: VecDeque::with_capacity(window_size),
            window_size,
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
            sma5_sum: 0.0,
            sma10_sum: 0.0,
            rsi_gains: VecDeque::with_capacity(14),
            rsi_losses: VecDeque::with_capacity(14),
            ema12: None,
            ema26: None,
            volatility_sum_sq: 0.0,
            volume_sma_sum: 0.0,
        }
    }

    fn add_trade(&mut self, price: f64, volume: f64) {
        let trade = TradeData { price, volume };
        
        // Update rolling statistics
        if !self.trades.is_empty() {
            let prev_price = self.trades.back().unwrap().price;
            let _prev_volume = self.trades.back().unwrap().volume;
            
            // Update SMA sums
            self.sma5_sum += price - if self.trades.len() >= 5 { self.trades[self.trades.len() - 5].price } else { 0.0 };
            self.sma10_sum += price - if self.trades.len() >= 10 { self.trades[self.trades.len() - 10].price } else { 0.0 };
            
            // Update RSI gains/losses
            let change = price - prev_price;
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { -change } else { 0.0 };
            
            self.rsi_gains.push_back(gain);
            self.rsi_losses.push_back(loss);
            
            if self.rsi_gains.len() > 14 {
                self.rsi_gains.pop_front();
                self.rsi_losses.pop_front();
            }
            
            // Update EMAs for MACD
            let multiplier12 = 2.0 / (12.0 + 1.0);
            let multiplier26 = 2.0 / (26.0 + 1.0);
            
            self.ema12 = Some(match self.ema12 {
                Some(ema) => (price - ema) * multiplier12 + ema,
                None => price,
            });
            
            self.ema26 = Some(match self.ema26 {
                Some(ema) => (price - ema) * multiplier26 + ema,
                None => price,
            });
            
            // Update volume SMA
            self.volume_sma_sum += volume - if self.trades.len() >= 5 { self.trades[self.trades.len() - 5].volume } else { 0.0 };
            
            // Update volatility (simplified rolling variance)
            if self.trades.len() >= 10 {
                let old_price = self.trades[self.trades.len() - 10].price;
                let current_sma = self.sma5_sum / 5.0; // Using SMA5 as approximation
                self.volatility_sum_sq += (price - current_sma).powi(2) - (old_price - current_sma).powi(2);
            }
        } else {
            // First trade - initialize
            self.sma5_sum = price;
            self.sma10_sum = price;
            self.ema12 = Some(price);
            self.ema26 = Some(price);
            self.volume_sma_sum = volume;
        }
        
        self.trades.push_back(trade);
        if self.trades.len() > self.window_size {
            self.trades.pop_front();
        }
        if self.trades.len() == 50 { // Need more trades for better training
            println!("Starting training with {} trades", self.trades.len());
            self.train_model("linear_regression");
        }
    }

    fn calculate_sma(&self, period: usize) -> Option<f64> {
        if self.trades.len() < period {
            return None;
        }
        match period {
            5 => Some(self.sma5_sum / 5.0),
            10 => Some(self.sma10_sum / 10.0),
            _ => {
                // Fallback for other periods
                let sum: f64 = self.trades.iter().rev().take(period).map(|t| t.price).sum();
                Some(sum / period as f64)
            }
        }
    }

    fn calculate_volatility(&self, periods: usize) -> Option<f64> {
        if self.trades.len() < periods {
            return None;
        }
        let prices: Vec<f64> = self.trades.iter().rev().take(periods).map(|t| t.price).collect();
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;
        Some(variance.sqrt())
    }

    fn calculate_rsi(&self, period: usize) -> Option<f64> {
        self.calculate_rsi_from_index(self.trades.len() - 1, period)
    }

    fn calculate_volume_sma(&self, periods: usize) -> Option<f64> {
        self.calculate_volume_sma_from_index(self.trades.len() - 1, periods)
    }

    fn calculate_momentum(&self, periods: usize) -> Option<f64> {
        self.calculate_momentum_from_index(self.trades.len() - 1, periods)
    }

    fn calculate_macd(&self) -> Option<f64> {
        self.calculate_macd_from_index(self.trades.len() - 1)
    }

    fn calculate_bollinger_upper(&self, period: usize, std_dev: f64) -> Option<f64> {
        self.calculate_bollinger_upper_from_index(self.trades.len() - 1, period, std_dev)
    }

    fn calculate_bollinger_lower(&self, period: usize, std_dev: f64) -> Option<f64> {
        self.calculate_bollinger_lower_from_index(self.trades.len() - 1, period, std_dev)
    }

    fn train_model(&mut self, model_type: &str) {
        println!("train_model called with type: {}", model_type);
        let n = self.trades.len();
        if n < 35 { // need more data for better training
            println!("Not enough trades for training: {}", n);
            return;
        }
        let mut features = Vec::new();
        let mut targets = Vec::new();
        for i in 10..n-4 { // predict 3 trades ahead
            let _price = self.trades[i].price;
            let _volume = self.trades[i].volume;
            // Use momentum-based features instead of absolute indicators
            let price_momentum = self.calculate_momentum_from_index(i-1, 3).unwrap_or(0.0); // 3-period momentum
            let volume_change = if i >= 2 { 
                (self.trades[i-1].volume - self.trades[i-2].volume) / self.trades[i-2].volume.max(1.0)
            } else { 0.0 };
            let sma_momentum = self.calculate_momentum_from_index(i-1, 5).unwrap_or(0.0); // SMA momentum
            
            features.push(vec![price_momentum, volume_change, sma_momentum]);
            let future_price = self.trades[i+3].price; // predict 3 trades ahead
            let current_price = self.trades[i].price;
            let price_change = (future_price - current_price) / current_price;
            targets.push(price_change);
        }
        if features.is_empty() {
            println!("No features collected for training");
            return;
        }
        
        println!("Collected {} training samples", features.len());
        println!("Sample features: {:?}", &features[0..3]);
        println!("Sample targets: {:.6}, {:.6}, {:.6}", targets[0], targets[1], targets[2]);

        // Calculate normalization parameters
        let num_features = features[0].len();
        let mut means = vec![0.0; num_features];
        let mut stds = vec![0.0; num_features];

        // Calculate means
        for feature_vec in &features {
            for (i, &val) in feature_vec.iter().enumerate() {
                means[i] += val;
            }
        }
        for mean in &mut means {
            *mean /= features.len() as f64;
        }

        // Calculate standard deviations
        for feature_vec in &features {
            for (i, &val) in feature_vec.iter().enumerate() {
                stds[i] += (val - means[i]).powi(2);
            }
        }
        for std in &mut stds {
            *std = (*std / features.len() as f64).sqrt();
            if *std == 0.0 {
                *std = 1.0; // Avoid division by zero
            }
        }

        // Normalize features
        let mut normalized_features = Vec::new();
        for feature_vec in &features {
            let mut normalized = Vec::new();
            for (i, &val) in feature_vec.iter().enumerate() {
                normalized.push((val - means[i]) / stds[i]);
            }
            normalized_features.push(normalized);
        }

        // Store normalization parameters
        self.feature_means = means;
        self.feature_stds = stds;

        match model_type {
            "random_forest" => {
                // Convert to smartcore format (DenseMatrix and Vec)
                match DenseMatrix::from_2d_array(&normalized_features.iter().map(|row| row.as_slice()).collect::<Vec<_>>()) {
                    Ok(x) => {
                        let y = targets.to_vec();
                        
                        match RandomForestRegressor::fit(&x, &y, Default::default()) {
                            Ok(model) => {
                                self.model = Some(MLModel::RandomForest(model));
                                println!("Random Forest model trained successfully with {} samples", x.shape().0);
                            }
                            Err(e) => {
                                eprintln!("Failed to train Random Forest model: {:?}", e);
                                // Fallback to linear regression
                                self.train_linear_regression(&normalized_features, &targets);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to create DenseMatrix: {:?}", e);
                        // Fallback to linear regression
                        self.train_linear_regression(&normalized_features, &targets);
                    }
                }
            }
            _ => {
                // Default to linear regression
                self.train_linear_regression(&normalized_features, &targets);
            }
        }
    }

    fn train_linear_regression(&mut self, normalized_features: &[Vec<f64>], targets: &[f64]) {
        let x = Array2::from_shape_vec((normalized_features.len(), normalized_features[0].len()), 
                                      normalized_features.iter().flatten().cloned().collect()).unwrap();
        let y = Array1::from_vec(targets.to_vec());
        let dataset = Dataset::new(x, y);
        match LinearRegression::default().fit(&dataset) {
            Ok(model) => {
                self.model = Some(MLModel::LinearRegression(model));
                println!("Linear Regression model trained successfully with {} samples", dataset.nsamples());
            }
            Err(e) => {
                println!("Linear Regression training failed: {:?}", e);
            }
        }
    }

    fn calculate_sma_from_index(&self, end_idx: usize, period: usize) -> Option<f64> {
        if end_idx < period - 1 {
            return None;
        }
        let sum: f64 = (0..period).map(|i| self.trades[end_idx - i].price).sum();
        Some(sum / period as f64)
    }

    fn calculate_momentum_from_index(&self, end_idx: usize, periods: usize) -> Option<f64> {
        if end_idx < periods {
            return None;
        }
        let current = self.trades[end_idx].price;
        let past = self.trades[end_idx - periods].price;
        Some((current - past) / past)
    }

    fn calculate_volatility_from_index(&self, end_idx: usize, periods: usize) -> Option<f64> {
        // If requesting volatility for the latest data, use rolling statistics
        if end_idx == self.trades.len() - 1 && periods == 10 && self.trades.len() >= 10 {
            let variance = self.volatility_sum_sq / 10.0;
            return Some(variance.sqrt());
        }
        
        // Fallback to original calculation for historical data or other periods
        if end_idx < periods - 1 {
            return None;
        }
        let prices: Vec<f64> = (0..periods).map(|i| self.trades[end_idx - i].price).collect();
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;
        Some(variance.sqrt())
    }

    fn calculate_volume_sma_from_index(&self, end_idx: usize, periods: usize) -> Option<f64> {
        // If requesting volume SMA for the latest data, use rolling statistics
        if end_idx == self.trades.len() - 1 && periods == 5 && self.trades.len() >= 5 {
            return Some(self.volume_sma_sum / 5.0);
        }
        
        // Fallback to original calculation for historical data or other periods
        if end_idx < periods - 1 {
            return None;
        }
        let sum: f64 = (0..periods).map(|i| self.trades[end_idx - i].volume).sum();
        Some(sum / periods as f64)
    }

    fn calculate_rsi_from_index(&self, end_idx: usize, period: usize) -> Option<f64> {
        // If requesting RSI for the latest data, use rolling statistics
        if end_idx == self.trades.len() - 1 && self.rsi_gains.len() >= period {
            let avg_gain: f64 = self.rsi_gains.iter().sum::<f64>() / period as f64;
            let avg_loss: f64 = self.rsi_losses.iter().sum::<f64>() / period as f64;
            if avg_loss == 0.0 {
                return Some(100.0);
            }
            let rs = avg_gain / avg_loss;
            return Some(100.0 - (100.0 / (1.0 + rs)));
        }
        
        // Fallback to original calculation for historical data
        if end_idx < period {
            return None;
        }
        let mut gains = 0.0;
        let mut losses = 0.0;
        for i in 1..=period {
            let change = self.trades[end_idx - i + 1].price - self.trades[end_idx - i].price;
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }
        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;
        if avg_loss == 0.0 {
            return Some(100.0);
        }
        let rs = avg_gain / avg_loss;
        Some(100.0 - (100.0 / (1.0 + rs)))
    }

    fn calculate_macd_from_index(&self, end_idx: usize) -> Option<f64> {
        // If requesting MACD for the latest data, use rolling EMAs
        if end_idx == self.trades.len() - 1 {
            if let (Some(ema12), Some(ema26)) = (self.ema12, self.ema26) {
                return Some(ema12 - ema26);
            }
        }
        
        // Fallback to original calculation for historical data
        if end_idx < 25 {
            return None;
        }
        let ema12 = self.calculate_ema_from_index(end_idx, 12).unwrap();
        let ema26 = self.calculate_ema_from_index(end_idx, 26).unwrap();
        Some(ema12 - ema26)
    }

    fn calculate_bollinger_upper_from_index(&self, end_idx: usize, period: usize, std_dev: f64) -> Option<f64> {
        if end_idx < period - 1 {
            return None;
        }
        let sma = self.calculate_sma_from_index(end_idx, period).unwrap();
        let variance = (0..period)
            .map(|i| {
                let price = self.trades[end_idx - i].price;
                (price - sma).powi(2)
            })
            .sum::<f64>() / period as f64;
        let std = variance.sqrt();
        Some(sma + std_dev * std)
    }

    fn calculate_bollinger_lower_from_index(&self, end_idx: usize, period: usize, std_dev: f64) -> Option<f64> {
        if end_idx < period - 1 {
            return None;
        }
        let sma = self.calculate_sma_from_index(end_idx, period).unwrap();
        let variance = (0..period)
            .map(|i| {
                let price = self.trades[end_idx - i].price;
                (price - sma).powi(2)
            })
            .sum::<f64>() / period as f64;
        let std = variance.sqrt();
        Some(sma - std_dev * std)
    }

    fn calculate_ema_from_index(&self, end_idx: usize, period: usize) -> Option<f64> {
        if end_idx < period - 1 || end_idx >= self.trades.len() {
            return None;
        }
        let multiplier = 2.0 / (period as f64 + 1.0);
        let start_idx = end_idx - period + 1;
        if start_idx >= self.trades.len() {
            return None;
        }
        let mut ema = self.trades[start_idx].price;
        for i in (start_idx + 1)..=end_idx {
            if i >= self.trades.len() {
                return None;
            }
            ema = (self.trades[i].price - ema) * multiplier + ema;
        }
        Some(ema)
    }

    fn predict_next(&self) -> Option<f64> {
        if let Some(model) = &self.model {
            if self.feature_means.is_empty() || self.feature_stds.is_empty() {
                return None;
            }
            let _last = self.trades.back()?;
            let second_last = &self.trades[self.trades.len() - 2];
            
            // Use momentum-based features for prediction
            let price_momentum = self.calculate_momentum_from_index(self.trades.len() - 2, 3).unwrap_or(0.0);
            let volume_change = if self.trades.len() >= 3 {
                let idx = self.trades.len() - 2;
                (self.trades[idx].volume - self.trades[idx-1].volume) / self.trades[idx-1].volume.max(1.0)
            } else { 0.0 };
            let sma_momentum = self.calculate_momentum_from_index(self.trades.len() - 2, 5).unwrap_or(0.0);
            
            // Raw features in same order as training
            let raw_features = vec![price_momentum, volume_change, sma_momentum];
            
            // Normalize features
            let mut normalized_features = Vec::new();
            for (i, &val) in raw_features.iter().enumerate() {
                if i < self.feature_means.len() && i < self.feature_stds.len() {
                    normalized_features.push((val - self.feature_means[i]) / self.feature_stds[i]);
                } else {
                    normalized_features.push(val); // Fallback if indices don't match
                }
            }
            
            match model {
                MLModel::LinearRegression(lr_model) => {
                    let input = Array2::from_shape_vec((1, 3), normalized_features).unwrap();
                    let predicted_change = lr_model.predict(&input)[0];
                    let current_price = self.trades.back().unwrap().price;
                    Some(current_price * (1.0 + predicted_change))
                }
                MLModel::RandomForest(rf_model) => {
                    match DenseMatrix::from_2d_array(&[normalized_features.as_slice()]) {
                        Ok(input) => {
                            rf_model.predict(&input).ok().and_then(|predictions| {
                                let predicted_change = predictions[0];
                                let current_price = self.trades.back().unwrap().price;
                                Some(current_price * (1.0 + predicted_change))
                            })
                        },
                        Err(_) => None,
                    }
                }
            }
        } else {
            None
        }
    }

    fn get_trading_signal(&self) -> Option<String> {
        let current = self.trades.back()?.price;
        let predicted = self.predict_next()?;
        let buy_threshold = 0.0005; // 0.05%
        let sell_threshold = 0.0005; // 0.05%
        let diff_pct = ((predicted - current) / current) * 100.0;
        
        // Debug output for first few predictions
        static mut COUNT: u32 = 0;
        unsafe {
            COUNT += 1;
            if COUNT <= 3 {
                println!("Prediction {}: Current={:.2}, Predicted={:.2}, Diff={:.4}%", COUNT, current, predicted, diff_pct);
            }
        }
        
        if predicted > current * (1.0 + buy_threshold) {
            Some("BUY".to_string())
        } else if predicted < current * (1.0 - sell_threshold) {
            Some("SELL".to_string())
        } else {
            Some("HOLD".to_string())
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    // Check for backtest mode
    let args: Vec<String> = env::args().collect();
    let is_backtest = args.contains(&"--backtest".to_string());

    if is_backtest {
        info!("Starting backtest mode");
        run_backtest().await?;
    } else {
        info!("Starting live trading bot demo with Binance and ML prediction");
        run_live().await?;
    }

    Ok(())
}

async fn run_live() -> Result<()> {
    info!("🚀 Starting LIVE TRADING BOT");
    info!("⚠️  WARNING: This will execute real trades!");
    info!("💰 Starting balance: $1000");
    info!("🎯 Risk management: 1% stop-loss, 2% take-profit, 10% position sizing");

    // Check for API keys (optional for now - will simulate)
    let api_key = env::var("BINANCE_API_KEY").ok();
    let secret_key = env::var("BINANCE_SECRET_KEY").ok();

    if api_key.is_none() || secret_key.is_none() {
        info!("⚠️  No API keys found - running in SIMULATION mode");
        info!("💡 Set BINANCE_API_KEY and BINANCE_SECRET_KEY for real trading");
    } else {
        info!("🔑 API keys found - ready for LIVE trading");
    }

    // Initialize market stream for BTC/USDT spot trades
    let mut streams = Streams::<PublicTrades>::builder()
        .subscribe([
            (BinanceSpot::default(), "btc", "usdt", MarketDataInstrumentKind::Spot, PublicTrades),
        ])
        .init()
        .await?;

    // Initialize live trader
    let mut trader = LiveTrader::new();

    // Select the stream for Binance
    let mut binance_stream = streams
        .select(barter_instrument::exchange::ExchangeId::BinanceSpot)
        .unwrap();

    info!("📡 Connected to Binance BTC/USDT stream");
    info!("🤖 Bot is now monitoring market and generating signals...");

    // Status update counter
    let mut status_counter = 0;
    let mut trade_count = 0;

    // Main trading loop
    while let Some(event) = binance_stream.next().await {
        match event {
            barter_data::streams::reconnect::Event::Item(result) => match result {
                Ok(trade) => {
                    let price = trade.kind.price;
                    let volume = trade.kind.amount;

                    // Process price update and check for trading signals
                    trader.process_price_update(price).await?;

                    trade_count += 1;
                    status_counter += 1;

                    // Print status every 100 trades
                    if status_counter >= 100 {
                        trader.print_status();
                        status_counter = 0;
                    }

                    // Safety check - stop after 1000 trades for demo
                    if trade_count >= 1000 {
                        info!("🛑 Demo limit reached (1000 trades) - stopping bot");
                        break;
                    }
                }
                Err(e) => {
                    warn!("❌ Error receiving trade: {:?}", e);
                    // Continue on errors
                }
            }
            _ => {}
        }
    }

    // Final status
    info!("🏁 Live trading session ended");
    trader.print_status();

    // Close any open position
    if trader.position > 0.0 {
        // In real trading, you'd get the current market price
        // For demo, we'll use a simulated close
        info!("💰 Closing remaining position...");
        // trader.execute_sell(current_price).await?;
    }

    Ok(())
}

async fn run_backtest() -> Result<()> {
    // Generate simulated trades with random walk
    let mut trades = Vec::new();
    let mut price = 50000.0; // starting price
    let base_volume = 1.0;
    
    for i in 0..1000 {
        // Random walk with small steps
        let change = (rand::random::<f64>() - 0.5) * 100.0; // +/- 50
        price += change;
        price = price.max(40000.0).min(60000.0); // bound it
        
        let volume = base_volume + (rand::random::<f64>() - 0.5) * 0.5;
        trades.push((price, volume));
    }

    // Initialize backtester
    let mut backtester = Backtester::new();

    // Now backtest on collected trades
    let prices: Vec<f64> = trades.iter().map(|(p, _)| *p).collect();
    let min_price = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_price = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    println!("Collected {} simulated trades, price range: {:.2} - {:.2}", trades.len(), min_price, max_price);
    
    for (price, amount) in &trades {
        backtester.process_trade(*price, *amount);
    }

    // Close any open position at market close
    if backtester.position > 0.0 {
        let last_price = trades.last().unwrap().0;
        backtester.sell(last_price);
        info!("Closed position at end of backtest at ${}", last_price);
    }

    // Print results
    backtester.print_results();

    Ok(())
}