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
    id: u64, // Unique ID for debugging
}

#[allow(dead_code)]
impl SimpleMLPredictor {
    pub fn new(window_size: usize) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
        println!("🆔 Created SimpleMLPredictor with ID: {}", id);
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
            id,
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
        println!("📊 Trade count: {} (training at 50)", self.trades.len());
        if self.trades.len() == 50 { // Need more trades for better training
            println!("🎯 Reached 50 trades - starting training");
            self.train_model("random_forest");
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
        println!("🔧 train_model called (ID: {}) with type: {}", self.id, model_type);
        println!("📊 Current trades: {}", self.trades.len());
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

            // Add advanced technical indicators
            let bollinger_position = self.calculate_bollinger_position_from_index(i-1, 20).unwrap_or(0.0);
            let stochastic = self.calculate_stochastic_from_index(i-1, 14, 3).unwrap_or(50.0);
            let williams_r = self.calculate_williams_r_from_index(i-1, 14).unwrap_or(-50.0);
            let volume_ratio = self.calculate_volume_ratio_from_index(i-1, 10).unwrap_or(1.0);
            let price_acceleration = self.calculate_price_acceleration_from_index(i-1, 5).unwrap_or(0.0);

            // Apply robust scaling and clipping to features during training
            let scaled_features = vec![
                self.clip_and_scale(price_momentum, -0.1, 0.1), // Price momentum: clip to ±10%
                self.clip_and_scale(volume_change, -2.0, 2.0),  // Volume change: clip to ±200%
                self.clip_and_scale(recent_volatility, 0.0, 0.05), // Volatility: clip to 0-5%
                rsi / 100.0, // RSI: already 0-1
                self.clip_and_scale(macd, -0.01, 0.01), // MACD: clip to reasonable range
                bollinger_position, // Already -1 to 1
                stochastic / 100.0, // Stochastic: 0-1
                (williams_r + 100.0) / 100.0, // Williams %R: 0-1
                self.clip_and_scale(volume_ratio, 0.1, 5.0), // Volume ratio: clip to 0.1-5.0
                self.clip_and_scale(price_acceleration, -0.01, 0.01), // Price acceleration: clip to ±1%
            ];
            
            features.push(scaled_features);
            let future_price = self.trades[i+3].price; // predict 3 trades ahead
            let current_price = self.trades[i].price;
            let price_change = (future_price - current_price) / current_price;

            // Improved target calculation: use significant price movements
            // Classify as BUY (1.0), SELL (-1.0), or HOLD (0.0) based on price change magnitude
            let target = if price_change.abs() < 0.003 { // Less than 0.3% change = HOLD (was 0.5%)
                0.0
            } else if price_change > 0.008 { // More than 0.8% gain = BUY signal (was 1%)
                1.0
            } else if price_change < -0.008 { // More than 0.8% loss = SELL signal (was 1%)
                -1.0
            } else {
                // Moderate changes (0.3% to 0.8%): use direction but weaker signal
                if price_change > 0.0 { 0.6 } else { -0.6 } // Stronger weak signals
            };
            targets.push(target);
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
        let has_variation = features.iter().any(|f| f.iter().any(|&v| v.abs() > 0.001)); // Require more significant variation
        let target_variation = targets.iter().any(|&t: &f64| t.abs() > 1e-9_f64); // Very low threshold to allow training even with tiny movements
        println!("🔍 Variation check: features={}, targets={}", has_variation, target_variation);
        if !has_variation || !target_variation {
            println!("Market too stable for ML training (features: {}, targets: {}) - switching to random trading mode for testing",
                    has_variation, target_variation);
            println!("Feature variation check: min |f| = {:.2e}, max |f| = {:.2e}",
                    features.iter().flatten().map(|&x| x.abs()).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0),
                    features.iter().flatten().map(|&x| x.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0));
            println!("Target variation check: min |t| = {:.2e}, max |t| = {:.2e}",
                    targets.iter().map(|&x: &f64| x.abs()).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0_f64),
                    targets.iter().map(|&x: &f64| x.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0_f64));
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
                        println!("✅ DenseMatrix created with shape: {:?}", x.shape());
                        let y = targets.to_vec();

                        // Create Random Forest with optimized parameters
                        match RandomForestRegressor::fit(&x, &y, smartcore::ensemble::random_forest_regressor::RandomForestRegressorParameters::default()
                            .with_n_trees(100)
                            .with_max_depth(10)
                            .with_min_samples_split(2)
                            .with_min_samples_leaf(1)) {
                            Ok(model) => {
                                println!("🎯 About to store Random Forest model...");
                                self.model = Some(MLModel::RandomForest(model));
                                println!("✅ Model assigned: {}", self.model.is_some());
                                println!("✅ Random Forest model trained successfully with {} samples!", x.shape().0);
                                println!("🎯 Model is now active - switching from random to ML-based trading");
                                println!("🔍 Model stored: {}", self.model.is_some());
                                println!("🔍 Model stored: {}", self.model.is_some());
                            }
                            Err(e) => {
                                println!("❌ Random Forest training failed: {:?}", e);
                                // Keep existing model or set to None
                            }
                        }
                    }
                    Err(e) => {
                        println!("❌ Failed to create DenseMatrix: {:?}", e);
                        // Keep existing model or set to None
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
            }
            Err(e) => {
                println!("❌ Linear Regression training failed: {:?}", e);
            }
        }
    }

    // Advanced Technical Indicators for Better ML Predictions

    fn calculate_bollinger_position_from_index(&self, index: usize, period: usize) -> Option<f64> {
        if index < period { return None; }
        
        let start = index.saturating_sub(period - 1);
        let prices: Vec<f64> = self.trades.iter().skip(start).take(period).map(|t| t.price).collect();
        
        let sma = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter().map(|p| (p - sma).powi(2)).sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();
        
        let upper_band = sma + (2.0 * std_dev);
        let lower_band = sma - (2.0 * std_dev);
        let current_price = self.trades[index].price;
        
        // Return position within bands: -1 (lower) to +1 (upper)
        if upper_band == lower_band {
            Some(0.0)
        } else {
            Some(2.0 * (current_price - lower_band) / (upper_band - lower_band) - 1.0)
        }
    }

    fn calculate_stochastic_from_index(&self, index: usize, k_period: usize, _d_period: usize) -> Option<f64> {
        if index < k_period { return None; }
        
        let start = index.saturating_sub(k_period - 1);
        let recent_prices: Vec<f64> = self.trades.iter().skip(start).take(k_period).map(|t| t.price).collect();
        
        let highest = recent_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = recent_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let current = self.trades[index].price;
        
        if highest == lowest {
            Some(50.0) // Neutral when no range
        } else {
            Some(100.0 * (current - lowest) / (highest - lowest))
        }
    }

    fn calculate_williams_r_from_index(&self, index: usize, period: usize) -> Option<f64> {
        if index < period { return None; }
        
        let start = index.saturating_sub(period - 1);
        let recent_prices: Vec<f64> = self.trades.iter().skip(start).take(period).map(|t| t.price).collect();
        
        let highest = recent_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = recent_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let current = self.trades[index].price;
        
        if highest == lowest {
            Some(-50.0) // Neutral
        } else {
            Some(-100.0 * (highest - current) / (highest - lowest))
        }
    }

    fn calculate_volume_ratio_from_index(&self, index: usize, period: usize) -> Option<f64> {
        if index < period { return None; }
        
        let start = index.saturating_sub(period - 1);
        let recent_volumes: Vec<f64> = self.trades.iter().skip(start).take(period).map(|t| t.volume).collect();
        let current_volume = self.trades[index].volume;
        
        let avg_volume = recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64;
        
        if avg_volume == 0.0 {
            Some(1.0)
        } else {
            Some(current_volume / avg_volume)
        }
    }

    fn calculate_price_acceleration_from_index(&self, index: usize, period: usize) -> Option<f64> {
        if index < period + 1 { return None; }
        
        // Calculate rate of change of momentum (acceleration)
        let current_momentum = self.calculate_momentum_from_index(index, period)?;
        let previous_momentum = self.calculate_momentum_from_index(index - 1, period)?;
        
        Some(current_momentum - previous_momentum)
    }

    fn calculate_rsi_from_index(&self, index: usize, period: usize) -> Option<f64> {
        if index < period { return None; }
        
        let start = index.saturating_sub(period - 1);
        let prices: Vec<f64> = self.trades.iter().skip(start).take(period).map(|t| t.price).collect();
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in 1..prices.len() {
            let change = prices[i] - prices[i-1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }
        
        if losses == 0.0 {
            Some(100.0)
        } else {
            let rs = gains / losses;
            Some(100.0 - (100.0 / (1.0 + rs)))
        }
    }

    fn calculate_volume_sma_from_index(&self, index: usize, periods: usize) -> Option<f64> {
        if index < periods { return None; }
        
        let start = index.saturating_sub(periods - 1);
        let volumes: Vec<f64> = self.trades.iter().skip(start).take(periods).map(|t| t.volume).collect();
        
        Some(volumes.iter().sum::<f64>() / volumes.len() as f64)
    }

    pub fn calculate_momentum_from_index(&self, index: usize, periods: usize) -> Option<f64> {
        if index < periods { return None; }
        
        let current_price = self.trades[index].price;
        let past_price = self.trades[index - periods].price;
        
        Some((current_price - past_price) / past_price)
    }

    fn calculate_macd_from_index(&self, index: usize) -> Option<f64> {
        if index < 26 { return None; }
        
        // Calculate EMAs
        let mut ema12;
        let mut ema26;
        
        let start = index.saturating_sub(25);
        let prices: Vec<f64> = self.trades.iter().skip(start).take(26).map(|t| t.price).collect();
        
        // Initialize EMAs with first price
        ema12 = prices[0];
        ema26 = prices[0];
        
        // Calculate EMAs
        let multiplier12 = 2.0 / (12.0 + 1.0);
        let multiplier26 = 2.0 / (26.0 + 1.0);
        
        for &price in &prices[1..] {
            ema12 = (price - ema12) * multiplier12 + ema12;
            ema26 = (price - ema26) * multiplier26 + ema26;
        }
        
        Some(ema12 - ema26)
    }

    fn calculate_bollinger_upper_from_index(&self, index: usize, period: usize, std_dev: f64) -> Option<f64> {
        if index < period { return None; }
        
        let start = index.saturating_sub(period - 1);
        let prices: Vec<f64> = self.trades.iter().skip(start).take(period).map(|t| t.price).collect();
        
        let sma = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter().map(|p| (p - sma).powi(2)).sum::<f64>() / prices.len() as f64;
        let std = variance.sqrt();
        
        Some(sma + (std_dev * std))
    }

    fn calculate_bollinger_lower_from_index(&self, index: usize, period: usize, std_dev: f64) -> Option<f64> {
        if index < period { return None; }
        
        let start = index.saturating_sub(period - 1);
        let prices: Vec<f64> = self.trades.iter().skip(start).take(period).map(|t| t.price).collect();
        
        let sma = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter().map(|p| (p - sma).powi(2)).sum::<f64>() / prices.len() as f64;
        let std = variance.sqrt();
        
        Some(sma - (std_dev * std))
    }

    fn calculate_volatility_from_index(&self, index: usize, periods: usize) -> Option<f64> {
        if index < periods { return None; }
        
        let start = index.saturating_sub(periods - 1);
        let prices: Vec<f64> = self.trades.iter().skip(start).take(periods).map(|t| t.price).collect();
        
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;
        
        Some(variance.sqrt())
    }

    pub fn predict_next(&self) -> Option<f64> {
        if self.model.is_none() || self.trades.len() < 10 {
            return None;
        }

        // Extract current features with improved scaling
        let price_momentum = self.calculate_momentum_from_index(self.trades.len() - 1, 5).unwrap_or(0.0);
        let volume_change = if self.trades.len() >= 2 {
            let current_vol = self.trades.back().unwrap().volume;
            let prev_vol = self.trades[self.trades.len() - 2].volume;
            (current_vol - prev_vol) / prev_vol.max(1.0)
        } else { 0.0 };
        let recent_volatility = self.calculate_volatility_from_index(self.trades.len() - 1, 10).unwrap_or(0.001);
        let rsi = self.calculate_rsi_from_index(self.trades.len() - 1, 14).unwrap_or(50.0);
        let macd = self.calculate_macd_from_index(self.trades.len() - 1).unwrap_or(0.0);
        
        // Add advanced indicators
        let bollinger_position = self.calculate_bollinger_position_from_index(self.trades.len() - 1, 20).unwrap_or(0.0);
        let stochastic = self.calculate_stochastic_from_index(self.trades.len() - 1, 14, 3).unwrap_or(50.0);
        let williams_r = self.calculate_williams_r_from_index(self.trades.len() - 1, 14).unwrap_or(-50.0);
        let volume_ratio = self.calculate_volume_ratio_from_index(self.trades.len() - 1, 10).unwrap_or(1.0);
        let price_acceleration = self.calculate_price_acceleration_from_index(self.trades.len() - 1, 5).unwrap_or(0.0);

        // Apply robust scaling and clipping to prevent outliers
        let features = [
            self.clip_and_scale(price_momentum, -0.1, 0.1), // Price momentum: clip to ±10%
            self.clip_and_scale(volume_change, -2.0, 2.0),  // Volume change: clip to ±200%
            self.clip_and_scale(recent_volatility, 0.0, 0.05), // Volatility: clip to 0-5%
            rsi / 100.0, // RSI: already 0-1
            self.clip_and_scale(macd, -0.01, 0.01), // MACD: clip to reasonable range
            bollinger_position, // Already -1 to 1
            stochastic / 100.0, // Stochastic: 0-1
            (williams_r + 100.0) / 100.0, // Williams %R: 0-1
            self.clip_and_scale(volume_ratio, 0.1, 5.0), // Volume ratio: clip to 0.1-5.0
            self.clip_and_scale(price_acceleration, -0.01, 0.01), // Price acceleration: clip to ±1%
        ];

        // Normalize features using stored parameters
        let mut normalized_features = Vec::new();
        for (i, &val) in features.iter().enumerate() {
            if i < self.feature_means.len() && i < self.feature_stds.len() && self.feature_stds[i] > 0.0 {
                let normalized = (val - self.feature_means[i]) / self.feature_stds[i];
                // Additional clipping of normalized values to prevent extreme outliers
                normalized_features.push(normalized.clamp(-3.0, 3.0));
            } else {
                normalized_features.push(val); // Fallback if normalization params not available
            }
        }

        // Make prediction
        match &self.model {
            Some(MLModel::RandomForest(model)) => {
                let input = DenseMatrix::from_2d_array(&[normalized_features.as_slice()]).ok()?;
                let prediction = model.predict(&input).ok()?;
                if !prediction.is_empty() {
                    Some(prediction[0])
                } else {
                    None
                }
            }
            Some(MLModel::LinearRegression(model)) => {
                let input = Array2::from_shape_vec((1, normalized_features.len()), normalized_features).ok()?;
                let prediction = model.predict(&input);
                prediction.get(0).copied()
            }
            None => None,
        }
    }

    /// Clip value to range and scale to improve feature distribution
    fn clip_and_scale(&self, value: f64, min_val: f64, max_val: f64) -> f64 {
        let clipped = value.max(min_val).min(max_val);
        // Apply tanh transformation for better distribution
        (clipped - (min_val + max_val) / 2.0) / ((max_val - min_val) / 2.0)
    }
}
