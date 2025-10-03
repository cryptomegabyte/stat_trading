use crate::types::{TradeData, MLModel};
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};
use std::collections::VecDeque;

#[derive(Debug)]
pub struct SimpleMLPredictor {
    pub model: Option<MLModel>,
    pub trades: VecDeque<TradeData>,
    pub window_size: usize,
    pub feature_means: Vec<f64>,
    pub feature_stds: Vec<f64>,
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

#[allow(dead_code)]
impl SimpleMLPredictor {
    pub fn new(window_size: usize) -> Self {
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

    pub fn add_trade(&mut self, price: f64, volume: f64) {
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
            println!("🎯 Reached 50 trades - starting training");
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

    pub fn calculate_volatility(&self, periods: usize) -> Option<f64> {
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

    pub fn train_model(&mut self, model_type: &str) {
        println!("🔧 train_model called with type: {}", model_type);
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
            // Use more robust features that work even in stable markets
            let price_momentum = self.calculate_momentum_from_index(i-1, 5).unwrap_or(0.0);
            let volume_change = if i >= 2 {
                (self.trades[i-1].volume - self.trades[i-2].volume) / self.trades[i-2].volume.max(1.0)
            } else { 0.0 };
            let recent_volatility = self.calculate_volatility_from_index(i-1, 10).unwrap_or(0.001);

            // Only use RSI and MACD if we have enough data
            let rsi = if i >= 14 { self.calculate_rsi_from_index(i-1, 14).unwrap_or(50.0) } else { 50.0 };
            let macd = if i >= 26 { self.calculate_macd_from_index(i-1).unwrap_or(0.0) } else { 0.0 };

            features.push(vec![price_momentum, volume_change, recent_volatility, rsi / 100.0, macd]);
            let future_price = self.trades[i+3].price; // predict 3 trades ahead
            let current_price = self.trades[i].price;
            let price_change = (future_price - current_price) / current_price;
            targets.push(price_change);
        }
        if features.is_empty() {
            println!("No features collected for training");
            return;
        }

        // Debug: Check raw data variation
        if self.trades.len() > 10 {
            println!("🔍 Debug training: Last 5 trades:");
            for i in (self.trades.len().saturating_sub(5)..self.trades.len()).rev() {
                if let Some(trade) = self.trades.get(i) {
                    println!("  Price: {:.2}, Volume: {:.6}", trade.price, trade.volume);
                }
            }
        }

        println!("Collected {} training samples", features.len());
        println!("Sample features: {:?}", &features[0..3]);
        println!("Sample targets: {:.6}, {:.6}, {:.6}", targets[0], targets[1], targets[2]);

        // Check if we have sufficient variation in features
        let has_variation = features.iter().any(|f| f.iter().any(|&v| v.abs() > 0.01)); // Require more significant variation
        let target_variation = targets.iter().any(|&t| t.abs() > 0.001); // Check if targets vary
        if !has_variation || !target_variation {
            println!("Market too stable for ML training (features: {}, targets: {}) - switching to random trading mode for testing",
                    has_variation, target_variation);
            // Don't train model, but allow random trades for testing infrastructure
            return;
        }

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
        println!("train_linear_regression called with {} samples", normalized_features.len());
        let x = Array2::from_shape_vec((normalized_features.len(), normalized_features[0].len()),
                                      normalized_features.iter().flatten().cloned().collect()).unwrap();
        let y = Array1::from_vec(targets.to_vec());
        let dataset = Dataset::new(x, y);
        match LinearRegression::default().fit(&dataset) {
            Ok(model) => {
                self.model = Some(MLModel::LinearRegression(model));
                println!("Linear Regression model trained successfully with {} samples", dataset.nsamples());
                println!("✅ Model is now set!");
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
        let start_idx = end_idx.saturating_sub(period - 1);
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

    pub fn predict_next(&self) -> Option<f64> {
        if let Some(model) = &self.model {
            if self.feature_means.is_empty() || self.feature_stds.is_empty() {
                return None;
            }
            let _last = self.trades.back()?;
            let _second_last = &self.trades[self.trades.len() - 2];

            // Debug: Check raw data
            if self.trades.len() > 10 {
                println!("🔍 Debug: Last 5 trades:");
                for i in (self.trades.len().saturating_sub(5)..self.trades.len()).rev() {
                    if let Some(trade) = self.trades.get(i) {
                        println!("  Price: {:.2}, Volume: {:.6}", trade.price, trade.volume);
                    }
                }
            }

            // Use same features as training for prediction
            let price_momentum = self.calculate_momentum_from_index(self.trades.len() - 2, 5).unwrap_or(0.0);
            let volume_change = if self.trades.len() >= 3 {
                let idx = self.trades.len() - 2;
                (self.trades[idx].volume - self.trades[idx-1].volume) / self.trades[idx-1].volume.max(1.0)
            } else { 0.0 };
            let recent_volatility = self.calculate_volatility_from_index(self.trades.len() - 2, 10).unwrap_or(0.001);

            // Only use RSI and MACD if we have enough data
            let rsi = if self.trades.len() >= 15 { self.calculate_rsi_from_index(self.trades.len() - 2, 14).unwrap_or(50.0) } else { 50.0 };
            let macd = if self.trades.len() >= 27 { self.calculate_macd_from_index(self.trades.len() - 2).unwrap_or(0.0) } else { 0.0 };

            // Raw features in same order as training
            let raw_features = [price_momentum, volume_change, recent_volatility, rsi / 100.0, macd];

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
                    let input = Array2::from_shape_vec((1, normalized_features.len()), normalized_features.clone()).unwrap();
                    let predicted_change = lr_model.predict(&input)[0];
                    let current_price = self.trades.back().unwrap().price;
                    Some(current_price * (1.0 + predicted_change))
                }
                MLModel::RandomForest(rf_model) => {
                    match DenseMatrix::from_2d_array(&[normalized_features.as_slice()]) {
                        Ok(input) => {
                            rf_model.predict(&input).ok().map(|predictions| {
                                let predicted_change = predictions[0];
                                let current_price = self.trades.back().unwrap().price;
                                current_price * (1.0 + predicted_change)
                            })
                        },
                        Err(_) => None,
                    }
                }
            }
        } else {
            // No trained model available - use random trading for testing infrastructure
            println!("🤖 No ML model available - using random trading for infrastructure testing");
            None
        }
    }

    pub fn get_trading_signal(&self) -> Option<String> {
        let current = self.trades.back()?.price;

        // Try to get ML prediction first
        if let Some(predicted) = self.predict_next() {
            let buy_threshold = 0.0001; // 0.01%
            let sell_threshold = 0.0001; // 0.01%
            let _diff_pct = ((predicted - current) / current) * 100.0;

            if predicted > current * (1.0 + buy_threshold) {
                Some("BUY".to_string())
            } else if predicted < current * (1.0 - sell_threshold) {
                Some("SELL".to_string())
            } else {
                Some("HOLD".to_string())
            }
        } else {
            // No ML model - use random trading for testing
            use std::time::{SystemTime, UNIX_EPOCH};
            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
            let random_signal = (timestamp % 100) < 20; // 20% chance of trade for testing

            if random_signal {
                // Return special signal for position-aware random trading
                Some("RANDOM".to_string())
            } else {
                Some("HOLD".to_string())
            }
        }
    }
}