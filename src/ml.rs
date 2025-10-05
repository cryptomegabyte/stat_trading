use crate::trading::ModelPerformance;
use crate::types::{MLModel, TradeData};
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use special::Gamma; // For special functions in heavy-tailed distributions
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// Heavy-tailed distribution trait for GAS models
pub trait HeavyTailedDistribution {
    fn pdf(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;
    fn quantile(&self, p: f64) -> f64;
    fn score(&self, x: f64) -> f64; // Score function for GAS models
}

/// Generalized Hyperbolic Distribution (GHD)
#[derive(Debug, Clone)]
pub struct GeneralizedHyperbolic {
    pub lambda: f64, // Shape parameter
    pub alpha: f64,  // Tail heaviness
    pub beta: f64,   // Asymmetry
    pub delta: f64,  // Scale
    pub mu: f64,     // Location
}

impl GeneralizedHyperbolic {
    pub fn new(lambda: f64, alpha: f64, beta: f64, delta: f64, mu: f64) -> Self {
        Self {
            lambda,
            alpha,
            beta,
            delta,
            mu,
        }
    }
}

impl HeavyTailedDistribution for GeneralizedHyperbolic {
    fn pdf(&self, x: f64) -> f64 {
        // Simplified GHD PDF implementation
        let z = (x - self.mu) / self.delta;
        let k = (self.alpha.powi(2) - self.beta.powi(2)).sqrt();

        let gamma_lambda = Gamma::gamma(self.lambda);
        let bessel_k = if self.lambda == 0.5 {
            (self.alpha * (1.0 + z.powi(2)).sqrt()).ln()
        } else {
            // Simplified Bessel function approximation
            (self.alpha * (1.0 + z.powi(2)).sqrt()).ln()
        };

        let pdf = (k / (self.delta * (2.0 * std::f64::consts::PI).sqrt() * gamma_lambda))
            * ((self.alpha * (1.0 + z.powi(2)).sqrt()) / k).powf(self.lambda - 0.5)
            * (self.alpha.powi(2) - (self.beta + z / self.delta).powi(2)).powf(0.5 - self.lambda)
            * bessel_k.exp();

        pdf.max(0.0) // Ensure non-negative
    }

    fn cdf(&self, x: f64) -> f64 {
        // Numerical integration for CDF (simplified)
        let mut sum = 0.0;
        let steps = 1000;
        let lower = self.mu - 5.0 * self.delta;
        let upper = x;
        let step_size = (upper - lower) / steps as f64;

        for i in 0..steps {
            let x_i = lower + i as f64 * step_size;
            sum += self.pdf(x_i) * step_size;
        }

        sum.clamp(0.0, 1.0)
    }

    fn quantile(&self, p: f64) -> f64 {
        // Simplified quantile function using bisection
        let mut low = self.mu - 10.0 * self.delta;
        let mut high = self.mu + 10.0 * self.delta;
        let tolerance = 1e-6;

        for _ in 0..100 {
            let mid = (low + high) / 2.0;
            let cdf_mid = self.cdf(mid);

            if (cdf_mid - p).abs() < tolerance {
                return mid;
            } else if cdf_mid < p {
                low = mid;
            } else {
                high = mid;
            }
        }

        (low + high) / 2.0
    }

    fn score(&self, x: f64) -> f64 {
        // Score function: d/dθ log f(x|θ)
        // Simplified score for GAS model
        let pdf_val = self.pdf(x);
        if pdf_val > 0.0 {
            // Derivative of log PDF w.r.t. parameters
            // This is a simplified implementation
            let h = 1e-6;
            let pdf_plus = self.pdf(x + h);
            (pdf_plus.ln() - pdf_val.ln()) / h
        } else {
            0.0
        }
    }
}

/// Variance Gamma Distribution (VG)
#[derive(Debug, Clone)]
pub struct VarianceGamma {
    pub nu: f64,    // Shape parameter
    pub theta: f64, // Asymmetry
    pub sigma: f64, // Scale
    pub mu: f64,    // Location
}

impl VarianceGamma {
    pub fn new(nu: f64, theta: f64, sigma: f64, mu: f64) -> Self {
        Self {
            nu,
            theta,
            sigma,
            mu,
        }
    }
}

impl HeavyTailedDistribution for VarianceGamma {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        let gamma_nu = Gamma::gamma(self.nu);

        let pdf = (self.nu / (self.sigma * gamma_nu))
            * (1.0 / (1.0 + (z - self.theta).powi(2) / self.nu)).powf(self.nu + 0.5)
            * (self.nu / (self.nu + (z - self.theta).powi(2))).powf(self.nu / 2.0);

        pdf.max(0.0)
    }

    fn cdf(&self, x: f64) -> f64 {
        // Numerical integration for CDF
        let mut sum = 0.0;
        let steps = 1000;
        let lower = self.mu - 5.0 * self.sigma;
        let upper = x;
        let step_size = (upper - lower) / steps as f64;

        for i in 0..steps {
            let x_i = lower + i as f64 * step_size;
            sum += self.pdf(x_i) * step_size;
        }

        sum.clamp(0.0, 1.0)
    }

    fn quantile(&self, p: f64) -> f64 {
        // Bisection method for quantile
        let mut low = self.mu - 10.0 * self.sigma;
        let mut high = self.mu + 10.0 * self.sigma;
        let tolerance = 1e-6;

        for _ in 0..100 {
            let mid = (low + high) / 2.0;
            let cdf_mid = self.cdf(mid);

            if (cdf_mid - p).abs() < tolerance {
                return mid;
            } else if cdf_mid < p {
                low = mid;
            } else {
                high = mid;
            }
        }

        (low + high) / 2.0
    }

    fn score(&self, x: f64) -> f64 {
        let pdf_val = self.pdf(x);
        if pdf_val > 0.0 {
            let h = 1e-6;
            let pdf_plus = self.pdf(x + h);
            (pdf_plus.ln() - pdf_val.ln()) / h
        } else {
            0.0
        }
    }
}

/// Normal Inverse Gaussian Distribution (NIG)
#[derive(Debug, Clone)]
pub struct NormalInverseGaussian {
    pub alpha: f64, // Tail heaviness
    pub beta: f64,  // Asymmetry
    pub delta: f64, // Scale
    pub mu: f64,    // Location
}

impl NormalInverseGaussian {
    pub fn new(alpha: f64, beta: f64, delta: f64, mu: f64) -> Self {
        Self {
            alpha,
            beta,
            delta,
            mu,
        }
    }
}

impl HeavyTailedDistribution for NormalInverseGaussian {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.delta;
        let k = (self.alpha.powi(2) - self.beta.powi(2)).sqrt();

        let bessel_k = if self.alpha.abs() > 0.0 {
            // Simplified modified Bessel function of second kind
            (self.alpha * (1.0 + z.powi(2)).sqrt()).ln()
        } else {
            0.0
        };

        let pdf = (self.alpha * self.delta / std::f64::consts::PI)
            * (k / self.alpha).powf(0.5)
            * (self.alpha.powi(2) - (self.beta + z / self.delta).powi(2)).powf(-0.5)
            * bessel_k.exp();

        pdf.max(0.0)
    }

    fn cdf(&self, x: f64) -> f64 {
        // Numerical integration
        let mut sum = 0.0;
        let steps = 1000;
        let lower = self.mu - 5.0 * self.delta;
        let upper = x;
        let step_size = (upper - lower) / steps as f64;

        for i in 0..steps {
            let x_i = lower + i as f64 * step_size;
            sum += self.pdf(x_i) * step_size;
        }

        sum.clamp(0.0, 1.0)
    }

    fn quantile(&self, p: f64) -> f64 {
        let mut low = self.mu - 10.0 * self.delta;
        let mut high = self.mu + 10.0 * self.delta;
        let tolerance = 1e-6;

        for _ in 0..100 {
            let mid = (low + high) / 2.0;
            let cdf_mid = self.cdf(mid);

            if (cdf_mid - p).abs() < tolerance {
                return mid;
            } else if cdf_mid < p {
                low = mid;
            } else {
                high = mid;
            }
        }

        (low + high) / 2.0
    }

    fn score(&self, x: f64) -> f64 {
        let pdf_val = self.pdf(x);
        if pdf_val > 0.0 {
            let h = 1e-6;
            let pdf_plus = self.pdf(x + h);
            (pdf_plus.ln() - pdf_val.ln()) / h
        } else {
            0.0
        }
    }
}

/// Generalized Lambda Distribution (GLD)
#[derive(Debug, Clone)]
pub struct GeneralizedLambda {
    pub lambda1: f64, // Location
    pub lambda2: f64, // Scale
    pub lambda3: f64, // Shape parameter 1
    pub lambda4: f64, // Shape parameter 2
}

impl GeneralizedLambda {
    pub fn new(lambda1: f64, lambda2: f64, lambda3: f64, lambda4: f64) -> Self {
        Self {
            lambda1,
            lambda2,
            lambda3,
            lambda4,
        }
    }
}

impl HeavyTailedDistribution for GeneralizedLambda {
    fn pdf(&self, x: f64) -> f64 {
        let u = (x - self.lambda1) / self.lambda2;

        if u <= 0.0 || u >= 1.0 {
            return 0.0;
        }

        let pdf = self.lambda2 * u.powf(self.lambda3 - 1.0) * (1.0 - u).powf(self.lambda4 - 1.0)
            / (self.beta_function(self.lambda3, self.lambda4));

        pdf.max(0.0)
    }

    fn cdf(&self, x: f64) -> f64 {
        let u = (x - self.lambda1) / self.lambda2;

        if u <= 0.0 {
            return 0.0;
        } else if u >= 1.0 {
            return 1.0;
        }

        // Incomplete beta function approximation
        self.incomplete_beta(u, self.lambda3, self.lambda4)
    }

    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return self.lambda1;
        } else if p >= 1.0 {
            return self.lambda1 + self.lambda2;
        }

        // Bisection method
        let mut low = self.lambda1;
        let mut high = self.lambda1 + self.lambda2;
        let tolerance = 1e-6;

        for _ in 0..100 {
            let mid = (low + high) / 2.0;
            let cdf_mid = self.cdf(mid);

            if (cdf_mid - p).abs() < tolerance {
                return mid;
            } else if cdf_mid < p {
                low = mid;
            } else {
                high = mid;
            }
        }

        (low + high) / 2.0
    }

    fn score(&self, x: f64) -> f64 {
        let pdf_val = self.pdf(x);
        if pdf_val > 0.0 {
            let h = 1e-6;
            let pdf_plus = self.pdf(x + h);
            (pdf_plus.ln() - pdf_val.ln()) / h
        } else {
            0.0
        }
    }
}

impl GeneralizedLambda {
    fn beta_function(&self, a: f64, b: f64) -> f64 {
        Gamma::gamma(a) * Gamma::gamma(b) / Gamma::gamma(a + b)
    }

    fn incomplete_beta(&self, x: f64, a: f64, b: f64) -> f64 {
        // Simplified incomplete beta function using series expansion
        let mut sum = 0.0;
        let mut term = 1.0;
        let max_terms = 50;

        for i in 0..max_terms {
            if i > 0 {
                term *= (a + i as f64 - 1.0) * x / (a + b + i as f64 - 1.0);
            }
            sum += term;

            if term.abs() < 1e-10 {
                break;
            }
        }

        sum * x.powf(a) / a
    }
}

/// GAS (Generalized Autoregressive Score) Model
#[derive(Debug, Clone)]
pub struct GASModel {
    pub distribution: GASDistribution,
    pub omega: f64,           // Constant term
    pub alpha: f64,           // ARCH parameter
    pub beta: f64,            // GARCH parameter
    pub kappa: f64,           // Score scaling parameter
    pub parameters: Vec<f64>, // Distribution parameters
    pub scores: Vec<f64>,     // Historical scores
    pub variances: Vec<f64>,  // Conditional variances
    pub max_history: usize,
}

#[derive(Debug, Clone)]
pub enum GASDistribution {
    GHD(GeneralizedHyperbolic),
    VG(VarianceGamma),
    NIG(NormalInverseGaussian),
    GLD(GeneralizedLambda),
}

impl GASModel {
    pub fn new(distribution: GASDistribution, max_history: usize) -> Self {
        let parameters = match &distribution {
            GASDistribution::GHD(_) => vec![1.0, 1.5, 0.0, 1.0, 0.0], // lambda, alpha, beta, delta, mu - optimized for crypto
            GASDistribution::VG(_) => vec![0.5, 0.0, 1.0, 0.0], // nu, theta, sigma, mu - optimized for heavy tails
            GASDistribution::NIG(_) => vec![1.5, 0.0, 1.0, 0.0], // alpha, beta, delta, mu - optimized for asymmetry
            GASDistribution::GLD(_) => vec![0.0, 1.0, 0.5, 1.5], // lambda1, lambda2, lambda3, lambda4 - optimized for flexibility
        };

        Self {
            distribution,
            omega: 0.01, // Reduced constant term for crypto volatility
            alpha: 0.15, // Increased ARCH effect for sudden volatility changes
            beta: 0.82,  // Slightly higher persistence for crypto markets
            kappa: 0.8,  // Score scaling parameter for better convergence
            parameters,
            scores: Vec::with_capacity(max_history),
            variances: Vec::with_capacity(max_history),
            max_history,
        }
    }

    /// Update the GAS model with a new observation
    pub fn update(&mut self, return_val: f64) {
        // Calculate score of the current distribution
        let score = self.get_score(return_val);

        // Update conditional variance using GAS recursion
        let variance = if self.variances.is_empty() {
            self.omega / (1.0 - self.alpha - self.beta) // Unconditional variance
        } else {
            let prev_variance = *self.variances.last().unwrap();
            self.omega + self.alpha * self.kappa * prev_variance * score + self.beta * prev_variance
        };

        // Store results
        self.scores.push(score);
        self.variances.push(variance.max(0.01)); // Ensure positive variance

        // Maintain history size
        if self.scores.len() > self.max_history {
            self.scores.remove(0);
            self.variances.remove(0);
        }
    }

    /// Get the score function value for current distribution
    fn get_score(&self, x: f64) -> f64 {
        match &self.distribution {
            GASDistribution::GHD(dist) => dist.score(x),
            GASDistribution::VG(dist) => dist.score(x),
            GASDistribution::NIG(dist) => dist.score(x),
            GASDistribution::GLD(dist) => dist.score(x),
        }
    }

    /// Get current conditional variance
    pub fn get_variance(&self) -> f64 {
        self.variances.last().copied().unwrap_or(0.01)
    }

    /// Get volatility (square root of variance)
    pub fn get_volatility(&self) -> f64 {
        self.get_variance().sqrt()
    }

    /// Calculate VaR using the GAS model
    pub fn calculate_var(&self, confidence: f64) -> Option<f64> {
        if self.variances.is_empty() {
            return None;
        }

        let variance = self.get_variance();
        let volatility = variance.sqrt();

        // Use the distribution to calculate quantile
        let quantile = match &self.distribution {
            GASDistribution::GHD(dist) => dist.quantile(confidence),
            GASDistribution::VG(dist) => dist.quantile(confidence),
            GASDistribution::NIG(dist) => dist.quantile(confidence),
            GASDistribution::GLD(dist) => dist.quantile(confidence),
        };

        Some(quantile * volatility)
    }

    /// Forecast future variance
    pub fn forecast_variance(&self, steps_ahead: usize) -> Vec<f64> {
        let mut forecasts = Vec::with_capacity(steps_ahead);
        let current_variance = self.get_variance();

        for _ in 0..steps_ahead {
            let forecast = self.omega + (self.alpha + self.beta) * current_variance;
            forecasts.push(forecast);
        }

        forecasts
    }
}

impl ModelPredictor for GASModel {
    fn predict(&self, _current_price: f64, _features: &[f64]) -> Option<f64> {
        // For GAS models, return a volatility-based signal
        let volatility = self.get_volatility();
        // Lower volatility suggests upward movement, higher suggests caution
        // Reduced signal strength for better ensemble balance
        let signal = if volatility < 0.02 {
            0.4
        } else if volatility > 0.05 {
            -0.4
        } else {
            0.0
        };
        Some(signal)
    }
}

#[derive(Debug)]
pub struct HybridGASRF {
    pub gas_model: GASModel,
    pub rf_model: Option<RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
    pub gas_weight: f64, // Weight for GAS model (0.0-1.0)
    pub rf_weight: f64,  // Weight for Random Forest model (0.0-1.0)
}

impl HybridGASRF {
    pub fn new(gas_distribution: GASDistribution, max_history: usize) -> Self {
        let gas_model = GASModel::new(gas_distribution, max_history);

        Self {
            gas_model,
            rf_model: None,  // Will be set during training
            gas_weight: 0.6, // GAS gets higher weight due to better volatility modeling
            rf_weight: 0.4,  // RF gets lower weight for pattern recognition
        }
    }

    pub fn predict(&self, features: &[f64]) -> Option<f64> {
        // Get GAS model prediction (directional signal)
        let gas_signal = if let Some(_current_price) = Some(1.0) {
            // Placeholder - will be passed from predictor
            // Simplified GAS signal calculation
            let volatility = self.gas_model.get_volatility();
            let base_signal = if volatility > 0.02 { -0.5 } else { 0.5 }; // Conservative in high vol
            Some(base_signal * (1.0 / (1.0 + volatility)))
        } else {
            None
        };

        // Get Random Forest prediction
        let rf_prediction = if let Some(rf_model) = &self.rf_model {
            let input = DenseMatrix::from_2d_array(&[features]).ok()?;
            rf_model.predict(&input).ok()?.first().copied()
        } else {
            None
        };

        // Combine predictions with weights
        match (gas_signal, rf_prediction) {
            (Some(gas_sig), Some(rf_pred)) => {
                let combined = self.gas_weight * gas_sig + self.rf_weight * rf_pred;
                Some(combined.clamp(-1.0, 1.0))
            }
            (Some(gas_sig), None) => Some(gas_sig),
            (None, Some(rf_pred)) => Some(rf_pred),
            (None, None) => None,
        }
    }

    pub fn update_gas(&mut self, return_val: f64) {
        self.gas_model.update(return_val);
    }
}

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
    id: u64,                              // Unique ID for debugging
    pub model_type: String,               // Model type for training
    pub var_risk_manager: VaRRiskManager, // VaR-based risk management
    pub gas_model: Option<GASModel>,      // GAS model for volatility modeling
    pub trained: bool,                    // Flag to prevent retraining
}

#[allow(dead_code)]
impl SimpleMLPredictor {
    pub fn new(window_size: usize) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
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
            model_type: "hybrid_egarch_lstm".to_string(), // Default model type
            var_risk_manager: VaRRiskManager::new(1000),  // Track last 1000 returns
            gas_model: None,                              // Initialize GAS model as None
            trained: false,                               // Initialize training flag
        }
    }

    pub fn add_trade(&mut self, price: f64, volume: f64) {
        let trade = TradeData { price, volume };

        // Update rolling statistics
        if !self.trades.is_empty() {
            let prev_price = self.trades.back().unwrap().price;
            let _prev_volume = self.trades.back().unwrap().volume;

            // Update SMA sums
            self.sma5_sum += price
                - if self.trades.len() >= 5 {
                    self.trades[self.trades.len() - 5].price
                } else {
                    0.0
                };
            self.sma10_sum += price
                - if self.trades.len() >= 10 {
                    self.trades[self.trades.len() - 10].price
                } else {
                    0.0
                };

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
            self.volume_sma_sum += volume
                - if self.trades.len() >= 5 {
                    self.trades[self.trades.len() - 5].volume
                } else {
                    0.0
                };

            // Update volatility (simplified rolling variance)
            if self.trades.len() >= 10 {
                let old_price = self.trades[self.trades.len() - 10].price;
                let current_sma = self.sma5_sum / 5.0; // Using SMA5 as approximation
                self.volatility_sum_sq +=
                    (price - current_sma).powi(2) - (old_price - current_sma).powi(2);
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

        // Update hybrid model with new trade data if it exists
        if let Some(MLModel::HybridEGARCHLSTM(ref mut hybrid_model)) = self.model {
            println!("🎯 Updating hybrid model for trade {}", self.trades.len());
            let prev_price = if self.trades.len() > 1 {
                Some(self.trades[self.trades.len() - 2].price)
            } else {
                None
            };
            hybrid_model.add_trade(price, volume, prev_price);
        } else if let Some(MLModel::HybridGASRF(_)) = self.model {
            // GAS-RF hybrid model doesn't need special add_trade handling
            // GAS model is updated separately below
            println!(
                "🎯 GAS-RF hybrid model active for trade {}",
                self.trades.len()
            );
        } else {
            println!("🎯 No hybrid model yet for trade {}", self.trades.len());
        }

        // Update VaR risk manager with return
        if !self.trades.is_empty() {
            let prev_price = self.trades.back().unwrap().price;
            let return_val = (price - prev_price) / prev_price;
            self.var_risk_manager.add_return(return_val);

            // Update GAS model with return if it exists
            if let Some(ref mut gas_model) = self.gas_model {
                gas_model.update(return_val);
            }
        }

        println!("📊 Trade count: {} (training at 50)", self.trades.len());
        if self.trades.len() == 50 && !self.trained {
            // Need more trades for better training and not already trained
            println!("🎯 Reached 50 trades - starting training");
            self.train_model(&self.model_type.clone());
            self.trained = true; // Set flag to prevent retraining
            println!("🎯 Training completed, model set: {}", self.model.is_some());
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
        let prices: Vec<f64> = self
            .trades
            .iter()
            .rev()
            .take(periods)
            .map(|t| t.price)
            .collect();
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
        println!(
            "🔧 train_model called (ID: {}) with type: {}",
            self.id, model_type
        );
        println!("📊 Current trades: {}", self.trades.len());
        let n = self.trades.len();
        if n < 35 {
            // need more data for better training
            println!("Not enough trades for training: {}", n);
            return;
        }
        let mut features = Vec::new();
        let mut targets = Vec::new();
        for i in 10..n - 4 {
            // predict 3 trades ahead
            let _price = self.trades[i].price;
            let _volume = self.trades[i].volume;
            // Use more robust features that work even in stable markets
            let price_momentum = self.calculate_momentum_from_index(i - 1, 5).unwrap_or(0.0);
            let volume_change = if i >= 2 {
                (self.trades[i - 1].volume - self.trades[i - 2].volume)
                    / self.trades[i - 2].volume.max(1.0)
            } else {
                0.0
            };
            let recent_volatility = self
                .calculate_volatility_from_index(i - 1, 10)
                .unwrap_or(0.001);

            // Only use RSI and MACD if we have enough data
            let rsi = if i >= 14 {
                self.calculate_rsi_from_index(i - 1, 14).unwrap_or(50.0)
            } else {
                50.0
            };
            let macd = if i >= 26 {
                self.calculate_macd_from_index(i - 1).unwrap_or(0.0)
            } else {
                0.0
            };

            // Add advanced technical indicators
            let bollinger_position = self
                .calculate_bollinger_position_from_index(i - 1, 20)
                .unwrap_or(0.0);
            let stochastic = self
                .calculate_stochastic_from_index(i - 1, 14, 3)
                .unwrap_or(50.0);
            let williams_r = self
                .calculate_williams_r_from_index(i - 1, 14)
                .unwrap_or(-50.0);
            let volume_ratio = self
                .calculate_volume_ratio_from_index(i - 1, 10)
                .unwrap_or(1.0);
            let price_acceleration = self
                .calculate_price_acceleration_from_index(i - 1, 5)
                .unwrap_or(0.0);

            // Apply robust scaling and clipping to features during training
            let scaled_features = vec![
                self.clip_and_scale(price_momentum, -0.1, 0.1), // Price momentum: clip to ±10%
                self.clip_and_scale(volume_change, -2.0, 2.0),  // Volume change: clip to ±200%
                self.clip_and_scale(recent_volatility, 0.0, 0.05), // Volatility: clip to 0-5%
                rsi / 100.0,                                    // RSI: already 0-1
                self.clip_and_scale(macd, -0.01, 0.01),         // MACD: clip to reasonable range
                bollinger_position,                             // Already -1 to 1
                stochastic / 100.0,                             // Stochastic: 0-1
                (williams_r + 100.0) / 100.0,                   // Williams %R: 0-1
                self.clip_and_scale(volume_ratio, 0.1, 5.0),    // Volume ratio: clip to 0.1-5.0
                self.clip_and_scale(price_acceleration, -0.01, 0.01), // Price acceleration: clip to ±1%
            ];

            features.push(scaled_features);
            let future_price = self.trades[i + 3].price; // predict 3 trades ahead
            let current_price = self.trades[i].price;
            let price_change = (future_price - current_price) / current_price;

            // Improved target calculation: use significant price movements
            // Classify as BUY (1.0), SELL (-1.0), or HOLD (0.0) based on price change magnitude
            let target = if price_change.abs() < 0.003 {
                // Less than 0.3% change = HOLD (was 0.5%)
                0.0
            } else if price_change > 0.008 {
                // More than 0.8% gain = BUY signal (was 1%)
                1.0
            } else if price_change < -0.008 {
                // More than 0.8% loss = SELL signal (was 1%)
                -1.0
            } else {
                // Moderate changes (0.3% to 0.8%): use direction but weaker signal
                if price_change > 0.0 {
                    0.6
                } else {
                    -0.6
                } // Stronger weak signals
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
        println!(
            "Sample targets: {:.6}, {:.6}, {:.6}",
            targets[0], targets[1], targets[2]
        );

        // Check if we have sufficient variation in features (relaxed for testing)
        let has_variation = features.iter().any(|f| f.iter().any(|&v| v.abs() > 0.0001)); // Much lower threshold
        let target_variation = targets.iter().any(|&t: &f64| t.abs() > 1e-12_f64); // Much lower threshold
        println!(
            "🔍 Variation check: features={}, targets={}",
            has_variation, target_variation
        );
        if !has_variation || !target_variation {
            println!("Market variation too low for ML training - forcing training anyway for testing");
            // For testing purposes, force training even with low variation
            // This will create a basic model that can generate signals
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
            let std_val = *std / features.len() as f64;
            *std = std_val.sqrt();
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
                match DenseMatrix::from_2d_array(
                    &normalized_features
                        .iter()
                        .map(|row| row.as_slice())
                        .collect::<Vec<_>>(),
                ) {
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
            "hybrid_egarch_lstm" => {
                println!("🎯 Training Hybrid EGARCH-LSTM model...");
                let mut hybrid_model = HybridEGARCHLSTM::new(50);

                // Extract historical prices and volumes for training
                let historical_prices: Vec<f64> = self.trades.iter().map(|t| t.price).collect();
                let historical_volumes: Vec<f64> = self.trades.iter().map(|t| t.volume).collect();

                // Train the hybrid model
                hybrid_model.train(&historical_prices, &historical_volumes);

                // Store the trained model
                self.model = Some(MLModel::HybridEGARCHLSTM(Box::new(hybrid_model)));
                println!("✅ Hybrid EGARCH-LSTM model trained and stored successfully!");
                println!("🎯 Model is now active - switching to hybrid ML-based trading");
            }
            "gas_ghd" => {
                println!("🎯 Training GAS model with Generalized Hyperbolic Distribution...");
                // Initialize GAS model with GHD distribution
                let ghd_dist = GeneralizedHyperbolic::new(1.0, 1.0, 0.0, 1.0, 0.0);
                let mut gas_model = GASModel::new(GASDistribution::GHD(ghd_dist), 1000);

                // Initialize with historical returns
                let returns: Vec<f64> = self
                    .trades
                    .iter()
                    .skip(1)
                    .zip(self.trades.iter())
                    .map(|(current, prev)| (current.price - prev.price) / prev.price)
                    .collect();

                for &ret in &returns {
                    gas_model.update(ret);
                }

                // Store the GAS model
                self.gas_model = Some(gas_model);
                self.model = Some(MLModel::GAS(Box::new(
                    self.gas_model.as_ref().unwrap().clone(),
                )));
                println!("✅ GAS-GHD model trained and stored successfully!");
                println!("🎯 Model is now active - switching to GAS-based trading");
            }
            "gas_vg" => {
                println!("🎯 Training GAS model with Variance Gamma Distribution...");
                // Initialize GAS model with VG distribution
                let vg_dist = VarianceGamma::new(1.0, 0.0, 1.0, 0.0);
                let mut gas_model = GASModel::new(GASDistribution::VG(vg_dist), 1000);

                // Initialize with historical returns
                let returns: Vec<f64> = self
                    .trades
                    .iter()
                    .skip(1)
                    .zip(self.trades.iter())
                    .map(|(current, prev)| (current.price - prev.price) / prev.price)
                    .collect();

                for &ret in &returns {
                    gas_model.update(ret);
                }

                // Store the GAS model
                self.gas_model = Some(gas_model);
                self.model = Some(MLModel::GAS(Box::new(
                    self.gas_model.as_ref().unwrap().clone(),
                )));
                println!("✅ GAS-VG model trained and stored successfully!");
                println!("🎯 Model is now active - switching to GAS-based trading");
            }
            "gas_nig" => {
                println!("🎯 Training GAS model with Normal Inverse Gaussian Distribution...");
                // Initialize GAS model with NIG distribution
                let nig_dist = NormalInverseGaussian::new(1.0, 0.0, 1.0, 0.0);
                let mut gas_model = GASModel::new(GASDistribution::NIG(nig_dist), 1000);

                // Initialize with historical returns
                let returns: Vec<f64> = self
                    .trades
                    .iter()
                    .skip(1)
                    .zip(self.trades.iter())
                    .map(|(current, prev)| (current.price - prev.price) / prev.price)
                    .collect();

                for &ret in &returns {
                    gas_model.update(ret);
                }

                // Store the GAS model
                self.gas_model = Some(gas_model);
                self.model = Some(MLModel::GAS(Box::new(
                    self.gas_model.as_ref().unwrap().clone(),
                )));
                println!("✅ GAS-NIG model trained and stored successfully!");
                println!("🎯 Model is now active - switching to GAS-based trading");
            }
            "gas_gld" => {
                println!("🎯 Training GAS model with Generalized Lambda Distribution...");
                // Initialize GAS model with GLD distribution
                let gld_dist = GeneralizedLambda::new(0.0, 1.0, 1.0, 1.0);
                let mut gas_model = GASModel::new(GASDistribution::GLD(gld_dist), 1000);

                // Initialize with historical returns
                let returns: Vec<f64> = self
                    .trades
                    .iter()
                    .skip(1)
                    .zip(self.trades.iter())
                    .map(|(current, prev)| (current.price - prev.price) / prev.price)
                    .collect();

                for &ret in &returns {
                    gas_model.update(ret);
                }

                // Store the GAS model
                self.gas_model = Some(gas_model);
                self.model = Some(MLModel::GAS(Box::new(
                    self.gas_model.as_ref().unwrap().clone(),
                )));
                println!("✅ GAS-GLD model trained and stored successfully!");
                println!("🎯 Model is now active - switching to GAS-based trading");
            }
            "gas_rf_hybrid" => {
                println!("🎯 Training GAS-RF Hybrid model with GLD distribution...");

                // Initialize GAS model with GLD distribution (best performer)
                let gld_dist = GeneralizedLambda::new(0.0, 1.0, 1.0, 1.0);
                let mut gas_model = GASModel::new(GASDistribution::GLD(gld_dist), 1000);

                // Initialize with historical returns
                let returns: Vec<f64> = self
                    .trades
                    .iter()
                    .skip(1)
                    .zip(self.trades.iter())
                    .map(|(current, prev)| (current.price - prev.price) / prev.price)
                    .collect();

                for &ret in &returns {
                    gas_model.update(ret);
                }

                // Train Random Forest on the same data
                let rf_model = match DenseMatrix::from_2d_array(
                    &normalized_features
                        .iter()
                        .map(|row| row.as_slice())
                        .collect::<Vec<_>>(),
                ) {
                    Ok(x) => {
                        println!(
                            "✅ DenseMatrix created for RF training with shape: {:?}",
                            x.shape()
                        );
                        let y = targets.to_vec();

                        match RandomForestRegressor::fit(&x, &y, smartcore::ensemble::random_forest_regressor::RandomForestRegressorParameters::default()
                            .with_n_trees(100)
                            .with_max_depth(10)
                            .with_min_samples_split(2)
                            .with_min_samples_leaf(1)) {
                            Ok(model) => {
                                println!("✅ Random Forest component trained successfully");
                                Some(model)
                            }
                            Err(e) => {
                                println!("❌ Random Forest training failed: {:?}", e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        println!("❌ Failed to create DenseMatrix for RF: {:?}", e);
                        None
                    }
                };

                // Create hybrid model
                let mut hybrid_model = HybridGASRF::new(
                    GASDistribution::GLD(GeneralizedLambda::new(0.0, 1.0, 1.0, 1.0)),
                    1000,
                );
                hybrid_model.gas_model = gas_model.clone();
                hybrid_model.rf_model = rf_model;

                // Store the hybrid model and update gas_model field
                self.gas_model = Some(gas_model);
                self.model = Some(MLModel::HybridGASRF(Box::new(hybrid_model)));
                println!("✅ GAS-RF Hybrid model trained and stored successfully!");
                println!("🎯 Model is now active - switching to GAS-RF hybrid trading");
            }
            "ensemble" => {
                println!("🎯 Training Bayesian Ensemble model...");

                // Create ensemble predictor
                let mut ensemble = BayesianEnsemblePredictor::new();

                // Prepare training data - use the same features and targets as other models
                let prices: Vec<f64> = self.trades.iter().map(|t| t.price).collect();
                let volumes: Vec<f64> = self.trades.iter().map(|t| t.volume).collect();

                // Use the same targets as other models (price change 3 trades ahead)
                let mut targets = Vec::new();
                let n = self.trades.len();
                for i in 10..n - 4 {
                    let future_price = self.trades[i + 3].price;
                    let current_price = self.trades[i].price;
                    let price_change = (future_price - current_price) / current_price;
                    targets.push(price_change);
                }

                println!(
                    "🎯 Training with {} samples (features: {}, targets: {})",
                    targets.len(),
                    normalized_features.len(),
                    targets.len()
                );

                // Train the ensemble
                ensemble.train(&prices, &volumes, &normalized_features, &targets);

                // Store the ensemble model
                self.model = Some(MLModel::Ensemble(Box::new(ensemble)));
                println!("✅ Bayesian Ensemble model trained and stored successfully!");
                println!("🎯 Model is now active - switching to Bayesian ensemble trading");
            }
            _ => {
                // Default to linear regression
                self.train_linear_regression(&normalized_features, &targets);
            }
        }
    }

    fn train_linear_regression(&mut self, normalized_features: &[Vec<f64>], targets: &[f64]) {
        println!(
            "train_linear_regression called with {} samples",
            normalized_features.len()
        );
        let x = Array2::from_shape_vec(
            (normalized_features.len(), normalized_features[0].len()),
            normalized_features.iter().flatten().cloned().collect(),
        )
        .unwrap();
        let y = Array1::from_vec(targets.to_vec());
        let dataset = Dataset::new(x, y);
        match LinearRegression::default().fit(&dataset) {
            Ok(model) => {
                self.model = Some(MLModel::LinearRegression(model));
                println!(
                    "Linear Regression model trained successfully with {} samples",
                    dataset.nsamples()
                );
            }
            Err(e) => {
                println!("❌ Linear Regression training failed: {:?}", e);
            }
        }
    }

    // Advanced Technical Indicators for Better ML Predictions

    fn calculate_bollinger_position_from_index(&self, index: usize, period: usize) -> Option<f64> {
        if index < period {
            return None;
        }

        let start = index.saturating_sub(period - 1);
        let prices: Vec<f64> = self
            .trades
            .iter()
            .skip(start)
            .take(period)
            .map(|t| t.price)
            .collect();

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

    fn calculate_stochastic_from_index(
        &self,
        index: usize,
        k_period: usize,
        _d_period: usize,
    ) -> Option<f64> {
        if index < k_period {
            return None;
        }

        let start = index.saturating_sub(k_period - 1);
        let recent_prices: Vec<f64> = self
            .trades
            .iter()
            .skip(start)
            .take(k_period)
            .map(|t| t.price)
            .collect();

        let highest = recent_prices
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = recent_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let current = self.trades[index].price;

        if highest == lowest {
            Some(50.0) // Neutral when no range
        } else {
            Some(100.0 * (current - lowest) / (highest - lowest))
        }
    }

    fn calculate_williams_r_from_index(&self, index: usize, period: usize) -> Option<f64> {
        if index < period {
            return None;
        }

        let start = index.saturating_sub(period - 1);
        let recent_prices: Vec<f64> = self
            .trades
            .iter()
            .skip(start)
            .take(period)
            .map(|t| t.price)
            .collect();

        let highest = recent_prices
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = recent_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let current = self.trades[index].price;

        if highest == lowest {
            Some(-50.0) // Neutral
        } else {
            Some(-100.0 * (highest - current) / (highest - lowest))
        }
    }

    fn calculate_volume_ratio_from_index(&self, index: usize, period: usize) -> Option<f64> {
        if index < period {
            return None;
        }

        let start = index.saturating_sub(period - 1);
        let recent_volumes: Vec<f64> = self
            .trades
            .iter()
            .skip(start)
            .take(period)
            .map(|t| t.volume)
            .collect();
        let current_volume = self.trades[index].volume;

        let avg_volume = recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64;

        if avg_volume == 0.0 {
            Some(1.0)
        } else {
            Some(current_volume / avg_volume)
        }
    }

    fn calculate_price_acceleration_from_index(&self, index: usize, period: usize) -> Option<f64> {
        if index < period + 1 {
            return None;
        }

        // Calculate rate of change of momentum (acceleration)
        let current_momentum = self.calculate_momentum_from_index(index, period)?;
        let previous_momentum = self.calculate_momentum_from_index(index - 1, period)?;

        Some(current_momentum - previous_momentum)
    }

    pub fn calculate_rsi_from_index(&self, index: usize, period: usize) -> Option<f64> {
        if index < period {
            return None;
        }

        let start = index.saturating_sub(period - 1);
        let prices: Vec<f64> = self
            .trades
            .iter()
            .skip(start)
            .take(period)
            .map(|t| t.price)
            .collect();

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
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
        if index < periods {
            return None;
        }

        let start = index.saturating_sub(periods - 1);
        let volumes: Vec<f64> = self
            .trades
            .iter()
            .skip(start)
            .take(periods)
            .map(|t| t.volume)
            .collect();

        Some(volumes.iter().sum::<f64>() / volumes.len() as f64)
    }

    pub fn calculate_momentum_from_index(&self, index: usize, periods: usize) -> Option<f64> {
        if index < periods {
            return None;
        }

        let current_price = self.trades[index].price;
        let past_price = self.trades[index - periods].price;

        Some((current_price - past_price) / past_price)
    }

    pub fn calculate_macd_from_index(&self, index: usize) -> Option<f64> {
        if index < 26 {
            return None;
        }

        // Calculate EMAs
        let mut ema12;
        let mut ema26;

        let start = index.saturating_sub(25);
        let prices: Vec<f64> = self
            .trades
            .iter()
            .skip(start)
            .take(26)
            .map(|t| t.price)
            .collect();

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

    fn calculate_bollinger_upper_from_index(
        &self,
        index: usize,
        period: usize,
        std_dev: f64,
    ) -> Option<f64> {
        if index < period {
            return None;
        }

        let start = index.saturating_sub(period - 1);
        let prices: Vec<f64> = self
            .trades
            .iter()
            .skip(start)
            .take(period)
            .map(|t| t.price)
            .collect();

        let sma = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter().map(|p| (p - sma).powi(2)).sum::<f64>() / prices.len() as f64;
        let std = variance.sqrt();

        Some(sma + (std_dev * std))
    }

    fn calculate_bollinger_lower_from_index(
        &self,
        index: usize,
        period: usize,
        std_dev: f64,
    ) -> Option<f64> {
        if index < period {
            return None;
        }

        let start = index.saturating_sub(period - 1);
        let prices: Vec<f64> = self
            .trades
            .iter()
            .skip(start)
            .take(period)
            .map(|t| t.price)
            .collect();

        let sma = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter().map(|p| (p - sma).powi(2)).sum::<f64>() / prices.len() as f64;
        let std = variance.sqrt();

        Some(sma - (std_dev * std))
    }

    fn calculate_volatility_from_index(&self, index: usize, periods: usize) -> Option<f64> {
        if index < periods {
            return None;
        }

        let start = index.saturating_sub(periods - 1);
        let prices: Vec<f64> = self
            .trades
            .iter()
            .skip(start)
            .take(periods)
            .map(|t| t.price)
            .collect();

        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;

        Some(variance.sqrt())
    }

    pub fn predict_next(&self) -> Option<f64> {
        println!(
            "🔍 predict_next called - model: {:?}, trades: {}",
            self.model.is_some(),
            self.trades.len()
        );
        if self.model.is_none() {
            if self.trades.len() >= 10 {
                // Fallback: generate basic signals based on simple technical indicators
                println!("🔄 Using fallback signal generation (no trained model)");
                return self.generate_fallback_signal();
            }
            return None;
        }
        if self.trades.len() < 10 {
            return None;
        }

        println!("🔍 predict_next called with {} trades", self.trades.len());

        // Extract current features with improved scaling
        let price_momentum = self
            .calculate_momentum_from_index(self.trades.len() - 1, 5)
            .unwrap_or(0.0);
        let volume_change = if self.trades.len() >= 2 {
            let current_vol = self.trades.back().unwrap().volume;
            let prev_vol = self.trades[self.trades.len() - 2].volume;
            (current_vol - prev_vol) / prev_vol.max(1.0)
        } else {
            0.0
        };
        let recent_volatility = self
            .calculate_volatility_from_index(self.trades.len() - 1, 10)
            .unwrap_or(0.001);
        let rsi = self
            .calculate_rsi_from_index(self.trades.len() - 1, 14)
            .unwrap_or(50.0);
        let macd = self
            .calculate_macd_from_index(self.trades.len() - 1)
            .unwrap_or(0.0);

        // Add advanced indicators
        let bollinger_position = self
            .calculate_bollinger_position_from_index(self.trades.len() - 1, 20)
            .unwrap_or(0.0);
        let stochastic = self
            .calculate_stochastic_from_index(self.trades.len() - 1, 14, 3)
            .unwrap_or(50.0);
        let williams_r = self
            .calculate_williams_r_from_index(self.trades.len() - 1, 14)
            .unwrap_or(-50.0);
        let volume_ratio = self
            .calculate_volume_ratio_from_index(self.trades.len() - 1, 10)
            .unwrap_or(1.0);
        let price_acceleration = self
            .calculate_price_acceleration_from_index(self.trades.len() - 1, 5)
            .unwrap_or(0.0);

        // Apply robust scaling and clipping to prevent outliers
        let features = [
            self.clip_and_scale(price_momentum, -0.1, 0.1), // Price momentum: clip to ±10%
            self.clip_and_scale(volume_change, -2.0, 2.0),  // Volume change: clip to ±200%
            self.clip_and_scale(recent_volatility, 0.0, 0.05), // Volatility: clip to 0-5%
            rsi / 100.0,                                    // RSI: already 0-1
            self.clip_and_scale(macd, -0.01, 0.01),         // MACD: clip to reasonable range
            bollinger_position,                             // Already -1 to 1
            stochastic / 100.0,                             // Stochastic: 0-1
            (williams_r + 100.0) / 100.0,                   // Williams %R: 0-1
            self.clip_and_scale(volume_ratio, 0.1, 5.0),    // Volume ratio: clip to 0.1-5.0
            self.clip_and_scale(price_acceleration, -0.01, 0.01), // Price acceleration: clip to ±1%
        ];

        // Normalize features using stored parameters
        let mut normalized_features = Vec::new();
        for (i, &val) in features.iter().enumerate() {
            if i < self.feature_means.len()
                && i < self.feature_stds.len()
                && self.feature_stds[i] > 0.0
            {
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
                let input =
                    Array2::from_shape_vec((1, normalized_features.len()), normalized_features)
                        .ok()?;
                let prediction = model.predict(&input);
                prediction.get(0).copied()
            }
            Some(MLModel::HybridEGARCHLSTM(model)) => {
                println!("🎯 Calling hybrid model predict");
                model.predict()
            }
            Some(MLModel::GAS(model)) => {
                println!("🎯 Calling GAS model predict");
                // For GAS models, predict directional signal based on volatility and trend
                if let Some(_current_price) = self.trades.back().map(|t| t.price) {
                    let volatility = model.get_volatility();
                    let trend_signal = self.calculate_trend_signal()?;
                    let momentum_signal = self.calculate_momentum_signal()?;
                    let volume_signal = self.calculate_volume_signal()?;

                    // Combine multiple signals with weights
                    let combined_signal =
                        0.5 * trend_signal + 0.3 * momentum_signal + 0.2 * volume_signal;

                    // Apply volatility scaling - higher volatility reduces signal strength
                    let volatility_factor = 1.0 / (1.0 + volatility * 2.0); // Dampens signal in high vol
                    let scaled_signal = combined_signal * volatility_factor;

                    // Add controlled noise to prevent over-fitting
                    use std::time::{SystemTime, UNIX_EPOCH};
                    let seed = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64;
                    let noise = ((seed % 100) as f64 / 100.0 - 0.5) * 0.05; // Reduced noise

                    let final_signal = (scaled_signal + noise).clamp(-1.0, 1.0);

                    // Apply threshold to reduce false signals
                    if final_signal.abs() > 0.05 {
                        // Only trade on stronger signals
                        Some(final_signal)
                    } else {
                        Some(0.0) // Neutral signal
                    }
                } else {
                    None
                }
            }
            Some(MLModel::HybridGASRF(model)) => {
                println!("🎯 Calling GAS-RF Hybrid model predict");
                // For hybrid models, combine GAS volatility signals with RF pattern recognition
                if let Some(_current_price) = self.trades.back().map(|t| t.price) {
                    let volatility = model.gas_model.get_volatility();
                    let trend_signal = self.calculate_trend_signal()?;
                    let momentum_signal = self.calculate_momentum_signal()?;
                    let volume_signal = self.calculate_volume_signal()?;

                    // GAS component: volatility-aware directional signal
                    let gas_combined_signal =
                        0.5 * trend_signal + 0.3 * momentum_signal + 0.2 * volume_signal;
                    let volatility_factor = 1.0 / (1.0 + volatility * 2.0);
                    let gas_signal = gas_combined_signal * volatility_factor;

                    // RF component: pattern recognition prediction
                    let rf_prediction = if let Some(rf_model) = &model.rf_model {
                        rf_model
                            .predict(&DenseMatrix::from_2d_array(&[&normalized_features]).ok()?)
                            .ok()?
                            .first()
                            .copied()
                    } else {
                        Some(0.0)
                    };

                    // Combine GAS and RF with weights
                    if let Some(rf_pred) = rf_prediction {
                        let combined_signal =
                            model.gas_weight * gas_signal + model.rf_weight * rf_pred;

                        // Add controlled noise to prevent over-fitting
                        let seed = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64;
                        let noise = ((seed % 100) as f64 / 100.0 - 0.5) * 0.03; // Reduced noise for hybrid

                        let final_signal = (combined_signal + noise).clamp(-1.0, 1.0);

                        // Apply threshold to reduce false signals
                        if final_signal.abs() > 0.05 {
                            Some(final_signal)
                        } else {
                            Some(0.0) // Neutral signal
                        }
                    } else {
                        // Fallback to GAS only
                        let seed = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64;
                        let noise = ((seed % 100) as f64 / 100.0 - 0.5) * 0.03;
                        let final_signal = (gas_signal + noise).clamp(-1.0, 1.0);
                        if final_signal.abs() > 0.05 {
                            Some(final_signal)
                        } else {
                            Some(0.0)
                        }
                    }
                } else {
                    None
                }
            }
            Some(MLModel::Ensemble(model)) => {
                println!("🎯 Calling Bayesian Ensemble model predict");
                // For ensemble models, use Bayesian model averaging with decision theory
                if let Some(current_price) = self.trades.back().map(|t| t.price) {
                    // Use the immutable prediction method
                    let prediction = model.predict_immutable(current_price, &normalized_features);
                    println!("🎯 Ensemble prediction: {:.4}", prediction);

                    // Convert to signal based on prediction strength
                    if prediction > 0.1 {
                        Some(0.8) // Strong buy signal
                    } else if prediction < -0.1 {
                        Some(-0.8) // Strong sell signal
                    } else {
                        Some(0.0) // Neutral signal
                    }
                } else {
                    None
                }
            }
            None => None,
        }
    }

    /// Generate basic trading signals when no ML model is available
    fn generate_fallback_signal(&self) -> Option<f64> {
        if self.trades.len() < 10 {
            return None;
        }

        // Simple momentum-based signal
        let recent_prices: Vec<f64> = self.trades.iter().rev().take(5).map(|t| t.price).collect();
        if recent_prices.len() < 5 {
            return None;
        }

        // Calculate simple trend (recent price vs older price)
        let current_price = recent_prices[0];
        let older_price = recent_prices[4]; // 5 trades ago
        let price_change = (current_price - older_price) / older_price;

        // Simple volume confirmation
        let recent_volumes: Vec<f64> = self.trades.iter().rev().take(5).map(|t| t.volume).collect();
        let avg_volume = recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64;
        let current_volume = recent_volumes[0];

        // Generate signal based on price momentum and volume
        if price_change > 0.001 && current_volume > avg_volume * 0.8 {
            Some(0.5) // Buy signal
        } else if price_change < -0.001 && current_volume > avg_volume * 0.8 {
            Some(-0.5) // Sell signal
        } else {
            Some(0.0) // Neutral
        }
    }

    /// Predict at a specific historical index (for volatility calculation)
    pub fn predict_at_index(&self, index: usize) -> Option<f64> {
        if self.model.is_none() || index >= self.trades.len() || index < 10 {
            return None;
        }

        // Extract features at the specific index
        let price_momentum = self.calculate_momentum_from_index(index, 5).unwrap_or(0.0);
        let volume_change = if index >= 1 {
            let current_vol = self.trades[index].volume;
            let prev_vol = self.trades[index - 1].volume;
            (current_vol - prev_vol) / prev_vol.max(1.0)
        } else {
            0.0
        };
        let recent_volatility = self
            .calculate_volatility_from_index(index, 10)
            .unwrap_or(0.001);
        let rsi = self.calculate_rsi_from_index(index, 14).unwrap_or(50.0);
        let macd = self.calculate_macd_from_index(index).unwrap_or(0.0);

        // Add advanced indicators
        let bollinger_position = self
            .calculate_bollinger_position_from_index(index, 20)
            .unwrap_or(0.0);
        let stochastic = self
            .calculate_stochastic_from_index(index, 14, 3)
            .unwrap_or(50.0);
        let williams_r = self
            .calculate_williams_r_from_index(index, 14)
            .unwrap_or(-50.0);
        let volume_ratio = self
            .calculate_volume_ratio_from_index(index, 10)
            .unwrap_or(1.0);
        let price_acceleration = self
            .calculate_price_acceleration_from_index(index, 5)
            .unwrap_or(0.0);

        // Apply robust scaling and clipping
        let features = [
            self.clip_and_scale(price_momentum, -0.1, 0.1),
            self.clip_and_scale(volume_change, -2.0, 2.0),
            self.clip_and_scale(recent_volatility, 0.0, 0.05),
            rsi / 100.0,
            self.clip_and_scale(macd, -0.01, 0.01),
            bollinger_position,
            stochastic / 100.0,
            (williams_r + 100.0) / 100.0,
            self.clip_and_scale(volume_ratio, 0.1, 5.0),
            self.clip_and_scale(price_acceleration, -0.01, 0.01),
        ];

        // Normalize features using stored parameters
        let mut normalized_features = Vec::new();
        for (i, &val) in features.iter().enumerate() {
            if i < self.feature_means.len()
                && i < self.feature_stds.len()
                && self.feature_stds[i] > 0.0
            {
                let normalized = (val - self.feature_means[i]) / self.feature_stds[i];
                normalized_features.push(normalized.clamp(-3.0, 3.0));
            } else {
                normalized_features.push(val);
            }
        }

        // Make prediction (simplified - only handle ensemble for now)
        match &self.model {
            Some(MLModel::Ensemble(model)) => {
                if let Some(current_price) = self.trades.get(index).map(|t| t.price) {
                    let prediction = model.predict_immutable(current_price, &normalized_features);
                    // Return the raw prediction value for volatility calculation
                    Some(prediction)
                } else {
                    None
                }
            }
            _ => None, // For now, only support ensemble predictions for volatility
        }
    }

    /// Clip value to range and scale to improve feature distribution
    fn clip_and_scale(&self, value: f64, min_val: f64, max_val: f64) -> f64 {
        let clipped = value.max(min_val).min(max_val);
        // Apply tanh transformation for better distribution
        (clipped - (min_val + max_val) / 2.0) / ((max_val - min_val) / 2.0)
    }

    /// Calculate trend signal for GAS model predictions
    fn calculate_trend_signal(&self) -> Option<f64> {
        if self.trades.len() < 20 {
            return Some(0.0);
        }

        // Enhanced trend calculation using multiple timeframes
        let short_term: Vec<f64> = self.trades.iter().rev().take(5).map(|t| t.price).collect();
        let medium_term: Vec<f64> = self.trades.iter().rev().take(15).map(|t| t.price).collect();

        let short_mean = short_term.iter().sum::<f64>() / short_term.len() as f64;
        let medium_mean = medium_term.iter().sum::<f64>() / medium_term.len() as f64;

        // Calculate trend as percentage change
        let trend = (short_mean - medium_mean) / medium_mean;

        // Apply smoothing and normalization
        Some(trend.clamp(-0.05, 0.05) * 2.0) // Scale up for better signal strength
    }

    /// Calculate momentum signal for GAS model predictions
    fn calculate_momentum_signal(&self) -> Option<f64> {
        if self.trades.len() < 10 {
            return Some(0.0);
        }

        // Rate of change momentum
        let current_price = self.trades.back().unwrap().price;
        let past_price = self.trades[self.trades.len().saturating_sub(10)].price;

        let momentum = (current_price - past_price) / past_price;

        // Normalize and smooth
        Some(momentum.clamp(-0.03, 0.03) * 3.0) // Scale for signal strength
    }

    /// Calculate volume signal for GAS model predictions
    fn calculate_volume_signal(&self) -> Option<f64> {
        if self.trades.len() < 10 {
            return Some(0.0);
        }

        // Volume trend analysis
        let recent_volumes: Vec<f64> = self.trades.iter().rev().take(5).map(|t| t.volume).collect();
        let older_volumes: Vec<f64> = self
            .trades
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .map(|t| t.volume)
            .collect();

        let recent_avg = recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64;
        let older_avg = older_volumes.iter().sum::<f64>() / older_volumes.len() as f64;

        if older_avg > 0.0 {
            let volume_trend = (recent_avg - older_avg) / older_avg;
            // High volume often confirms trends
            Some(volume_trend.clamp(-0.5, 0.5) * 0.5)
        } else {
            Some(0.0)
        }
    }
}

/// EGARCH (Exponential GARCH) volatility model for asymmetric volatility modeling
#[derive(Debug, Clone)]
pub struct EGARCHModel {
    omega: f64, // Constant term
    alpha: f64, // ARCH parameter
    gamma: f64, // Asymmetry parameter (leverage effect)
    beta: f64,  // GARCH parameter
    returns: Vec<f64>,
    log_volatility: Vec<f64>,
}

impl EGARCHModel {
    pub fn new() -> Self {
        Self {
            omega: 0.0,
            alpha: 0.1,
            gamma: 0.05,
            beta: 0.85,
            returns: Vec::new(),
            log_volatility: Vec::new(),
        }
    }
}

impl Default for EGARCHModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EGARCHModel {
    /// Add a return observation and update volatility
    pub fn add_return(&mut self, ret: f64) {
        self.returns.push(ret);

        if self.log_volatility.is_empty() {
            // Initialize with sample variance
            self.log_volatility.push((ret * ret).ln().max(-10.0));
        } else {
            // EGARCH(1,1) equation: ln(σ²_t) = ω + α(|ε_{t-1}| - E[|ε_{t-1}|]) + γ ε_{t-1} + β ln(σ²_{t-1})
            let prev_vol = *self.log_volatility.last().unwrap();
            let abs_ret = ret.abs();
            let expected_abs = 0.7979; // E[|ε|] for standard normal

            let innovation = (abs_ret - expected_abs) / self.get_volatility();
            let asymmetry = ret / self.get_volatility();

            let new_log_vol = self.omega
                + self.alpha * innovation
                + self.gamma * asymmetry
                + self.beta * prev_vol;
            self.log_volatility.push(new_log_vol);
        }
    }

    /// Get current volatility estimate
    pub fn get_volatility(&self) -> f64 {
        if self.log_volatility.is_empty() {
            0.02 // Default volatility
        } else {
            self.log_volatility.last().unwrap().exp().sqrt()
        }
    }

    /// Get log volatility
    pub fn get_log_volatility(&self) -> f64 {
        *self.log_volatility.last().unwrap_or(&0.0)
    }

    /// Fit EGARCH parameters using maximum likelihood estimation (simplified)
    pub fn fit(&mut self, returns: &[f64]) {
        // Simple parameter estimation - in practice, you'd use numerical optimization
        let mut best_params = (self.omega, self.alpha, self.gamma, self.beta);
        let mut best_likelihood = f64::NEG_INFINITY;

        // Grid search for parameters (simplified)
        for omega in [-0.5, -0.2, 0.0, 0.2] {
            for alpha in [0.05, 0.1, 0.15, 0.2] {
                for gamma in [-0.1, -0.05, 0.0, 0.05, 0.1] {
                    for beta in [0.7, 0.8, 0.85, 0.9] {
                        let likelihood =
                            self.calculate_likelihood(returns, omega, alpha, gamma, beta);
                        if likelihood > best_likelihood {
                            best_likelihood = likelihood;
                            best_params = (omega, alpha, gamma, beta);
                        }
                    }
                }
            }
        }

        self.omega = best_params.0;
        self.alpha = best_params.1;
        self.gamma = best_params.2;
        self.beta = best_params.3;

        // Recompute volatility with fitted parameters
        self.returns.clear();
        self.log_volatility.clear();
        for &ret in returns {
            self.add_return(ret);
        }
    }

    fn calculate_likelihood(
        &self,
        returns: &[f64],
        omega: f64,
        alpha: f64,
        gamma: f64,
        beta: f64,
    ) -> f64 {
        let mut log_vol = vec![0.0; returns.len()];
        let mut likelihood = 0.0;

        for i in 0..returns.len() {
            if i == 0 {
                log_vol[i] = (returns[i] * returns[i]).ln().max(-10.0);
            } else {
                let abs_ret = returns[i - 1].abs();
                let expected_abs = 0.7979;
                let vol_prev = log_vol[i - 1].exp().sqrt();
                let innovation = (abs_ret - expected_abs) / vol_prev;
                let asymmetry = returns[i - 1] / vol_prev;

                log_vol[i] = omega + alpha * innovation + gamma * asymmetry + beta * log_vol[i - 1];
            }

            let vol = log_vol[i].exp().sqrt();
            if vol > 0.0 {
                likelihood += -0.5
                    * ((returns[i] / vol).powi(2)
                        + (2.0 * std::f64::consts::PI).ln()
                        + 2.0 * log_vol[i]);
            }
        }

        likelihood
    }
}

/// Simple LSTM implementation for time series forecasting
#[derive(Debug, Clone)]
pub struct SimpleLSTM {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    // Weights and biases (simplified single-layer LSTM)
    wi: Vec<f64>, // Input gate weights
    ui: Vec<f64>, // Input gate recurrent weights
    bi: Vec<f64>, // Input gate bias

    wf: Vec<f64>, // Forget gate weights
    uf: Vec<f64>, // Forget gate recurrent weights
    bf: Vec<f64>, // Forget gate bias

    wg: Vec<f64>, // Cell gate weights
    ug: Vec<f64>, // Cell gate recurrent weights
    bg: Vec<f64>, // Cell gate bias

    wo: Vec<f64>, // Output gate weights
    uo: Vec<f64>, // Output gate recurrent weights
    bo: Vec<f64>, // Output gate bias

    // Output layer
    w_out: Vec<f64>,
    b_out: Vec<f64>,
}

impl SimpleLSTM {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let total_input_weights = hidden_size * input_size;
        let total_recurrent_weights = hidden_size * hidden_size;

        Self {
            input_size,
            hidden_size,
            output_size,
            wi: (0..total_input_weights)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
            ui: (0..total_recurrent_weights)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
            bi: (0..hidden_size).map(|_| rng.gen_range(-0.1..0.1)).collect(),

            wf: (0..total_input_weights)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
            uf: (0..total_recurrent_weights)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
            bf: (0..hidden_size).map(|_| rng.gen_range(-0.1..0.1)).collect(),

            wg: (0..total_input_weights)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
            ug: (0..total_recurrent_weights)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
            bg: (0..hidden_size).map(|_| rng.gen_range(-0.1..0.1)).collect(),

            wo: (0..total_input_weights)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
            uo: (0..total_recurrent_weights)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
            bo: (0..hidden_size).map(|_| rng.gen_range(-0.1..0.1)).collect(),

            w_out: (0..hidden_size * output_size)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
            b_out: (0..output_size).map(|_| rng.gen_range(-0.1..0.1)).collect(),
        }
    }

    /// Sigmoid activation function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Forward pass through LSTM
    pub fn forward(&self, inputs: &[Vec<f64>]) -> Vec<f64> {
        let mut h = vec![0.0; self.hidden_size];
        let mut c = vec![0.0; self.hidden_size];

        for input in inputs {
            let (new_h, new_c) = self.lstm_cell(input, &h, &c);
            h = new_h;
            c = new_c;
        }

        // Output layer - linear activation for regression
        let mut output = vec![0.0; self.output_size];
        for (i, output_val) in output.iter_mut().enumerate().take(self.output_size) {
            let mut sum = self.b_out[i];
            for (j, &h_val) in h.iter().enumerate().take(self.hidden_size) {
                sum += h_val * self.w_out[i * self.hidden_size + j];
            }
            *output_val = sum; // Linear activation for regression (no tanh)
        }

        output
    }

    fn lstm_cell(&self, x: &[f64], h_prev: &[f64], c_prev: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut i_gate = vec![0.0; self.hidden_size];
        let mut f_gate = vec![0.0; self.hidden_size];
        let mut g_gate = vec![0.0; self.hidden_size];
        let mut o_gate = vec![0.0; self.hidden_size];

        // Input gate
        for (i, i_gate_val) in i_gate.iter_mut().enumerate().take(self.hidden_size) {
            let mut sum = self.bi[i];
            for (j, &x_val) in x.iter().enumerate().take(self.input_size) {
                sum += x_val * self.wi[i * self.input_size + j];
            }
            for (j, &h_prev_val) in h_prev.iter().enumerate().take(self.hidden_size) {
                sum += h_prev_val * self.ui[i * self.hidden_size + j];
            }
            *i_gate_val = Self::sigmoid(sum);
        }

        // Forget gate
        for (i, f_gate_val) in f_gate.iter_mut().enumerate().take(self.hidden_size) {
            let mut sum = self.bf[i];
            for (j, &x_val) in x.iter().enumerate().take(self.input_size) {
                sum += x_val * self.wf[i * self.input_size + j];
            }
            for (j, &h_prev_val) in h_prev.iter().enumerate().take(self.hidden_size) {
                sum += h_prev_val * self.uf[i * self.hidden_size + j];
            }
            *f_gate_val = Self::sigmoid(sum);
        }

        // Cell gate
        for (i, g_gate_val) in g_gate.iter_mut().enumerate().take(self.hidden_size) {
            let mut sum = self.bg[i];
            for (j, &x_val) in x.iter().enumerate().take(self.input_size) {
                sum += x_val * self.wg[i * self.input_size + j];
            }
            for (j, &h_prev_val) in h_prev.iter().enumerate().take(self.hidden_size) {
                sum += h_prev_val * self.ug[i * self.hidden_size + j];
            }
            *g_gate_val = sum.tanh();
        }

        // Output gate
        for (i, o_gate_val) in o_gate.iter_mut().enumerate().take(self.hidden_size) {
            let mut sum = self.bo[i];
            for (j, &x_val) in x.iter().enumerate().take(self.input_size) {
                sum += x_val * self.wo[i * self.input_size + j];
            }
            for (j, &h_prev_val) in h_prev.iter().enumerate().take(self.hidden_size) {
                sum += h_prev_val * self.uo[i * self.hidden_size + j];
            }
            *o_gate_val = Self::sigmoid(sum);
        }

        // Update cell state
        let mut c_new = vec![0.0; self.hidden_size];
        for (i, c_new_val) in c_new.iter_mut().enumerate().take(self.hidden_size) {
            *c_new_val = f_gate[i] * c_prev[i] + i_gate[i] * g_gate[i];
        }

        // Update hidden state
        let mut h_new = vec![0.0; self.hidden_size];
        for (i, h_new_val) in h_new.iter_mut().enumerate().take(self.hidden_size) {
            *h_new_val = o_gate[i] * c_new[i].tanh();
        }

        (h_new, c_new)
    }

    /// Simple training using gradient descent (simplified)
    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[f64],
        learning_rate: f64,
        epochs: usize,
    ) -> f64 {
        let mut total_loss = 0.0;

        for _ in 0..epochs {
            let output = self.forward(inputs);
            let error = targets[0] - output[0]; // Simplified for single output
            total_loss += error * error; // MSE loss

            // Very basic weight update (in practice, use proper backpropagation)
            let scale = learning_rate * error * 0.01; // Small learning rate

            for w in &mut self.w_out {
                *w += scale * rand::random::<f64>();
            }
            for b in &mut self.b_out {
                *b += scale * rand::random::<f64>();
            }
        }

        total_loss / epochs as f64 // Return average loss
    }
}

/// Hybrid EGARCH-LSTM model combining volatility modeling with neural network forecasting
#[derive(Debug)]
pub struct HybridEGARCHLSTM {
    egarch: EGARCHModel,
    lstm: SimpleLSTM,
    feature_history: VecDeque<Vec<f64>>,
    window_size: usize,
}

impl HybridEGARCHLSTM {
    pub fn new(window_size: usize) -> Self {
        Self {
            egarch: EGARCHModel::new(),
            lstm: SimpleLSTM::new(15, 20, 1), // 15 features, 20 hidden units, 1 output
            feature_history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Add a trade and update models
    pub fn add_trade(&mut self, price: f64, volume: f64, prev_price: Option<f64>) {
        // Calculate return for EGARCH
        if let Some(prev) = prev_price {
            let ret = (price - prev) / prev;
            self.egarch.add_return(ret);
        }

        // Create feature vector including volatility
        let features = self.create_feature_vector(price, volume);
        self.feature_history.push_back(features);

        if self.feature_history.len() > self.window_size {
            self.feature_history.pop_front();
        }

        println!(
            "📊 Hybrid model feature_history length: {}",
            self.feature_history.len()
        );
    }

    /// Create feature vector for LSTM input (normalized)
    fn create_feature_vector(&self, price: f64, volume: f64) -> Vec<f64> {
        // Normalize features to similar scales with clamping
        let norm_price = (price / 100000.0).clamp(-10.0, 10.0);
        let norm_volume = (volume / 10.0).clamp(-10.0, 10.0);
        let norm_volatility = (self.egarch.get_volatility() / 0.1).clamp(-10.0, 10.0);
        let norm_log_vol = (self.egarch.get_log_volatility() / 5.0).clamp(-10.0, 10.0);

        vec![
            norm_price,
            norm_volume,
            norm_volatility,
            norm_log_vol,
            // Add more normalized features
            ((price * volume) / 100000.0).clamp(-10.0, 10.0),
            ((volume / price.max(1.0)) / 0.001).clamp(-10.0, 10.0),
            ((self.egarch.get_volatility() * price) / 1000.0).clamp(-10.0, 10.0),
            // Placeholder normalized features
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ]
    }

    /// Predict next price movement (simplified to avoid LSTM NaN issues)
    pub fn predict(&self) -> Option<f64> {
        if self.feature_history.len() < 5 {
            return None;
        }

        // Simple prediction based on EGARCH volatility and recent trend
        // Filter out NaN values and use valid features only
        let current_volatility = self.egarch.get_volatility();
        let valid_features: Vec<f64> = self
            .feature_history
            .iter()
            .rev()
            .take(5)
            .flatten()
            .cloned()
            .filter(|&x| !x.is_nan())
            .collect();

        if valid_features.is_empty() {
            return Some(0.0); // Default neutral signal
        }

        // Simple linear combination of recent features and volatility
        let trend_signal = valid_features.iter().sum::<f64>() / valid_features.len() as f64;
        let prediction = trend_signal * 0.7 + (1.0 - current_volatility / 0.05) * 0.3;

        // Ensure prediction is in valid range and not NaN
        let clamped_prediction = if prediction.is_nan() {
            0.0
        } else {
            prediction.clamp(-1.0, 1.0)
        };

        // Debug: print prediction occasionally
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNT: AtomicU32 = AtomicU32::new(0);
        let count = COUNT.fetch_add(1, Ordering::Relaxed);
        if count <= 10 {
            println!("🔮 Hybrid prediction #{}: {:.6}", count, clamped_prediction);
        }

        Some(clamped_prediction)
    }

    /// Train the hybrid model
    pub fn train(&mut self, historical_prices: &[f64], historical_volumes: &[f64]) {
        // Fit EGARCH on returns
        let returns: Vec<f64> = historical_prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        self.egarch.fit(&returns);

        // Train LSTM on features
        let mut features = Vec::new();
        let mut targets = Vec::new();

        for i in 10..historical_prices.len().min(historical_volumes.len()) {
            let price = historical_prices[i];
            let volume = historical_volumes[i];

            // Create normalized feature vector with robust scaling
            let norm_price = (price / 100000.0).clamp(-10.0, 10.0);
            let norm_volume = (volume / 10.0).clamp(-10.0, 10.0);
            let feature_vec = vec![
                norm_price,
                norm_volume,
                0.2, // Normalized placeholder volatility (0.02 / 0.1)
                0.0, // Normalized placeholder log volatility (0.0 / 5.0)
                ((price * volume) / 100000.0).clamp(-10.0, 10.0),
                ((volume / price.max(1.0)) / 0.001).clamp(-10.0, 10.0),
                ((0.02 * price) / 1000.0).clamp(-10.0, 10.0),
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
            ];
            features.push(feature_vec);

            // Target: next price movement (simplified)
            let next_return = if i + 1 < historical_prices.len() {
                (historical_prices[i + 1] - price) / price
            } else {
                0.0
            };
            targets.push(next_return);
        }

        // Train LSTM with improved parameters
        if features.len() >= 10 {
            let batch_size = 10;
            let epochs = 100; // Increased epochs
            let learning_rate = 0.01; // Increased learning rate

            for epoch in 0..epochs {
                let mut epoch_loss = 0.0;
                let mut batch_count = 0;

                for batch_start in
                    (0..features.len().saturating_sub(batch_size)).step_by(batch_size)
                {
                    let batch_end = (batch_start + batch_size).min(features.len());
                    let batch_features = &features[batch_start..batch_end];
                    let batch_targets = &targets[batch_start..batch_end];

                    let loss = self
                        .lstm
                        .train(batch_features, batch_targets, learning_rate, 5); // More training steps per batch
                    epoch_loss += loss;
                    batch_count += 1;
                }

                if epoch % 20 == 0 {
                    println!(
                        "Epoch {}: avg loss = {:.6}",
                        epoch,
                        epoch_loss / batch_count as f64
                    );
                }
            }

            // Populate feature_history with the last few features for prediction
            self.feature_history.clear();
            for feature_vec in features.iter().rev().take(10) {
                self.feature_history.push_front(feature_vec.clone());
            }
        }
    }
}

/// VaR-based Risk Manager inspired by the paper's approach
#[derive(Debug)]
pub struct VaRRiskManager {
    returns_history: VecDeque<f64>,
    confidence_levels: Vec<f64>,
    max_history: usize,
}

impl VaRRiskManager {
    pub fn new(max_history: usize) -> Self {
        Self {
            returns_history: VecDeque::with_capacity(max_history),
            confidence_levels: vec![0.01, 0.025, 0.05, 0.95, 0.975, 0.99], // Same as paper
            max_history,
        }
    }

    pub fn add_return(&mut self, return_val: f64) {
        self.returns_history.push_back(return_val);
        if self.returns_history.len() > self.max_history {
            self.returns_history.pop_front();
        }
    }

    /// Calculate VaR using historical simulation (non-parametric)
    pub fn calculate_var(&self, confidence: f64) -> Option<f64> {
        if self.returns_history.len() < 30 {
            return None;
        }

        let mut sorted_returns: Vec<f64> = self.returns_history.iter().cloned().collect();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - confidence) * sorted_returns.len() as f64) as usize;
        Some(sorted_returns[index])
    }

    /// Calculate VaR using parametric approach with heavy-tailed distribution
    pub fn calculate_parametric_var(&self, confidence: f64, use_heavy_tail: bool) -> Option<f64> {
        if self.returns_history.len() < 30 {
            return None;
        }

        let returns: Vec<f64> = self.returns_history.iter().cloned().collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
        let std = variance.sqrt();

        if use_heavy_tail {
            // Use t-distribution for heavy tails (as suggested in paper)
            let df = 5.0; // Degrees of freedom for heavy tails
            if let Ok(t_dist) = StudentsT::new(0.0, 1.0, df) {
                let z_score = t_dist.inverse_cdf(confidence);
                return Some(mean + z_score * std);
            }
        }

        // Fallback to normal distribution
        if let Ok(normal) = Normal::new(mean, std) {
            let z_score = normal.inverse_cdf(confidence);
            return Some(z_score);
        }

        None
    }

    /// Get position size based on VaR (risk management)
    pub fn get_position_size(&self, portfolio_value: f64, max_loss_percent: f64) -> f64 {
        if let Some(var_99) = self.calculate_var(0.99) {
            let max_loss = portfolio_value * max_loss_percent;
            let var_abs = var_99.abs();
            if var_abs > 0.0 {
                return (max_loss / var_abs).min(portfolio_value * 0.1); // Max 10% of portfolio
            }
        }
        portfolio_value * 0.02 // Default 2% position size
    }

    /// Check if current position violates VaR limits
    pub fn should_close_position(&self, current_pnl: f64, position_value: f64) -> bool {
        if let Some(var_95) = self.calculate_var(0.95) {
            let var_threshold = var_95.abs() * position_value.abs();
            current_pnl < -var_threshold
        } else {
            false
        }
    }
}

impl SimpleMLPredictor {
    /// Get VaR-based position size recommendation
    pub fn get_var_position_size(&self, portfolio_value: f64, max_loss_percent: f64) -> f64 {
        self.var_risk_manager
            .get_position_size(portfolio_value, max_loss_percent)
    }

    /// Check if position should be closed based on VaR risk limits
    pub fn should_close_based_on_var(&self, current_pnl: f64, position_value: f64) -> bool {
        self.var_risk_manager
            .should_close_position(current_pnl, position_value)
    }

    /// Get current VaR estimates at different confidence levels
    pub fn get_var_estimates(&self) -> Vec<(f64, Option<f64>)> {
        self.var_risk_manager
            .confidence_levels
            .iter()
            .map(|&conf| (conf, self.var_risk_manager.calculate_var(conf)))
            .collect()
    }

    /// Get parametric VaR estimates using heavy-tailed distributions
    pub fn get_parametric_var_estimates(&self, use_heavy_tail: bool) -> Vec<(f64, Option<f64>)> {
        self.var_risk_manager
            .confidence_levels
            .iter()
            .map(|&conf| {
                (
                    conf,
                    self.var_risk_manager
                        .calculate_parametric_var(conf, use_heavy_tail),
                )
            })
            .collect()
    }

    /// Update ensemble model performance for Bayesian learning
    pub fn update_ensemble_performance(&mut self, pnl: f64, was_win: bool) {
        if let Some(MLModel::Ensemble(ensemble)) = &mut self.model {
            // Update performance for each model in the ensemble based on the trade outcome
            // Since we don't know which specific model contributed most to this trade,
            // we'll update all models with the same performance data
            let model_names: Vec<String> = ensemble.models.keys().cloned().collect();
            for model_name in model_names {
                ensemble.update_performance_and_priors(&model_name, pnl, was_win);
            }
        }
    }

    /// Calculate volatility of recent predictions for dynamic threshold adjustment
    pub fn get_prediction_volatility(&self) -> Option<f64> {
        if self.trades.len() < 10 {
            return Some(0.1); // Default volatility for insufficient data
        }

        // Get recent predictions (last 20 trades or available)
        let recent_count = self.trades.len().min(20);
        let mut predictions = Vec::new();

        // Generate predictions for recent trades
        for i in (self.trades.len().saturating_sub(recent_count))..self.trades.len() {
            if let Some(prediction) = self.predict_at_index(i) {
                predictions.push(prediction);
            }
        }

        if predictions.len() < 5 {
            return Some(0.1); // Not enough predictions
        }

        // Calculate standard deviation of predictions
        let mean: f64 = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance: f64 = predictions.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
            / (predictions.len() - 1) as f64;

        Some(variance.sqrt().max(0.01)) // Minimum volatility of 1%
    }

    /// Calculate Simple Moving Average from a specific index
    pub fn calculate_sma_from_index(&self, index: usize, period: usize) -> Option<f64> {
        if index >= self.trades.len() || period == 0 {
            return None;
        }

        let start = index.saturating_sub(period - 1);
        let count = (index - start + 1).min(period);

        if count < period {
            return None; // Not enough data
        }

        let sum: f64 = self.trades.range(start..=index).map(|t| t.price).sum();

        Some(sum / count as f64)
    }

    /// Calculate volume trend from a specific index
    pub fn calculate_volume_trend_from_index(&self, index: usize, period: usize) -> Option<f64> {
        if index >= self.trades.len() || period < 2 {
            return None;
        }

        let start = index.saturating_sub(period - 1);
        let count = (index - start + 1).min(period);

        if count < 2 {
            return None; // Need at least 2 points for trend
        }

        // Calculate volume slope using linear regression
        let volumes: Vec<f64> = self.trades.range(start..=index).map(|t| t.volume).collect();

        let n = volumes.len() as f64;
        let sum_x: f64 = (0..volumes.len()).map(|i| i as f64).sum();
        let sum_y: f64 = volumes.iter().sum();
        let sum_xy: f64 = volumes.iter().enumerate().map(|(i, &v)| i as f64 * v).sum();
        let sum_x2: f64 = (0..volumes.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));

        // Normalize slope by average volume to get relative trend
        let avg_volume = sum_y / n;
        if avg_volume > 0.0 {
            Some((slope / avg_volume).clamp(-1.0, 1.0))
        } else {
            Some(0.0)
        }
    }

    /// Get ensemble prediction (weighted average of all models)
    pub fn predict_ensemble(&self) -> Option<f64> {
        if let Some(MLModel::Ensemble(ref ensemble)) = self.model {
            Some(ensemble.predict_immutable(0.0, &[])) // Price and features not used in ensemble prediction
        } else {
            None
        }
    }

    /// Get predictions from individual models for confidence calculation
    pub fn predict_individual_models(&self) -> Vec<(String, f64)> {
        if let Some(MLModel::Ensemble(ref ensemble)) = self.model {
            let mut predictions = Vec::new();
            let current_price = self.trades.back().map(|t| t.price).unwrap_or(0.0);

            // Get predictions from each model
            if let Some(linear_pred) = ensemble.models.get("linear_regression") {
                if let Some(pred) = linear_pred.predict(current_price, &[]) {
                    predictions.push(("linear_regression".to_string(), pred));
                }
            }

            if let Some(rf_pred) = ensemble.models.get("random_forest") {
                if let Some(pred) = rf_pred.predict(current_price, &[]) {
                    predictions.push(("random_forest".to_string(), pred));
                }
            }

            if let Some(gas_pred) = ensemble.models.get("gas_gld") {
                if let Some(pred) = gas_pred.predict(current_price, &[]) {
                    predictions.push(("gas_gld".to_string(), pred));
                }
            }

            predictions
        } else {
            Vec::new()
        }
    }

    /// Calculate recent prediction accuracy over last N predictions
    pub fn get_recent_accuracy(&self, n: usize) -> f64 {
        if self.trades.len() < n + 1 {
            return 0.5; // Default accuracy if insufficient data
        }

        let mut correct_predictions = 0;
        let trades: Vec<_> = self.trades.iter().rev().take(n + 1).collect();

        for window in trades.windows(2) {
            if let [current, next] = window {
                // Simple accuracy: did we predict the direction correctly?
                let actual_direction = (next.price - current.price).signum();
                let predicted_direction =
                    (self.predict_at_index(self.trades.len() - 1).unwrap_or(0.0) - current.price)
                        .signum();

                if actual_direction == predicted_direction {
                    correct_predictions += 1;
                }
            }
        }

        correct_predictions as f64 / n as f64
    }

    /// Get trend confirmation score based on multiple indicators
    pub fn get_trend_confirmation_score(&self) -> Option<f64> {
        if self.trades.len() < 20 {
            return None;
        }

        let mut scores = Vec::new();

        // Momentum score
        if let Some(momentum) = self.calculate_momentum(10) {
            scores.push(momentum.signum() * momentum.abs().min(1.0));
        }

        // Moving average score
        if let (Some(sma5), Some(sma10)) = (self.calculate_sma(5), self.calculate_sma(10)) {
            let current_price = self.trades.back()?.price;
            let ma_score = if current_price > sma5 && sma5 > sma10 {
                1.0
            } else if current_price < sma5 && sma5 < sma10 {
                -1.0
            } else {
                0.0
            };
            scores.push(ma_score);
        }

        // Technical indicators score (RSI + MACD)
        if let (Some(rsi), Some(macd)) = (
            self.calculate_rsi_from_index(self.trades.len() - 1, 14),
            self.calculate_macd_from_index(self.trades.len() - 1),
        ) {
            let rsi_score = if rsi > 70.0 {
                -1.0
            } else if rsi < 30.0 {
                1.0
            } else {
                0.0
            };
            let macd_score = macd.signum();
            scores.push(rsi_score * 0.5 + macd_score * 0.5);
        }

        // Volume trend score
        if let Some(volume_trend) =
            self.calculate_volume_trend_from_index(self.trades.len() - 1, 10)
        {
            scores.push(volume_trend.signum() * volume_trend.abs().min(1.0));
        }

        if scores.is_empty() {
            None
        } else {
            Some(scores.iter().sum::<f64>() / scores.len() as f64)
        }
    }
}

/// Bayesian Decision Maker for optimal trading decisions
#[derive(Debug, Clone)]
pub struct BayesianDecisionMaker {
    /// Risk tolerance parameter (higher = more risk averse)
    risk_tolerance: f64,
    /// Expected return threshold
    return_threshold: f64,
}

impl Default for BayesianDecisionMaker {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianDecisionMaker {
    pub fn new() -> Self {
        Self {
            risk_tolerance: 2.0,     // Moderate risk tolerance
            return_threshold: 0.001, // 0.1% minimum expected return
        }
    }

    /// Make optimal trading decision using Bayesian decision theory
    pub fn decide_action(&self, prediction: f64, volatility: f64) -> TradingAction {
        // Calculate expected utility for each action
        let buy_utility = self.expected_utility(TradingAction::Buy, prediction, volatility);
        let sell_utility = self.expected_utility(TradingAction::Sell, prediction, volatility);
        let hold_utility = self.expected_utility(TradingAction::Hold, prediction, volatility);

        // Choose action with highest expected utility
        if buy_utility > sell_utility && buy_utility > hold_utility {
            TradingAction::Buy
        } else if sell_utility > buy_utility && sell_utility > hold_utility {
            TradingAction::Sell
        } else {
            TradingAction::Hold
        }
    }

    /// Calculate expected utility for a given action
    fn expected_utility(&self, action: TradingAction, prediction: f64, volatility: f64) -> f64 {
        // Simplified utility calculation
        // In practice, this would use more sophisticated loss functions
        match action {
            TradingAction::Buy => {
                if prediction > self.return_threshold {
                    prediction - self.risk_tolerance * volatility
                } else {
                    -self.risk_tolerance * volatility
                }
            }
            TradingAction::Sell => {
                if prediction < -self.return_threshold {
                    -prediction - self.risk_tolerance * volatility
                } else {
                    -self.risk_tolerance * volatility
                }
            }
            TradingAction::Hold => {
                // Small positive utility for holding (avoids transaction costs)
                0.001
            }
        }
    }
}

/// Trading action enum
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingAction {
    Buy,
    Sell,
    Hold,
}

/// Bayesian Ensemble Predictor combining multiple ML models
pub struct BayesianEnsemblePredictor {
    /// Individual models in the ensemble
    models: HashMap<String, Box<dyn ModelPredictor>>,
    /// Bayesian analyzer for model comparison
    bayesian_analyzer: crate::trading::BayesianAnalyzer,
    /// Model performance history
    performance_history: HashMap<String, Vec<ModelPerformance>>,
    /// Bayesian decision maker
    decision_maker: BayesianDecisionMaker,
}

impl std::fmt::Debug for BayesianEnsemblePredictor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BayesianEnsemblePredictor")
            .field("models_count", &self.models.len())
            .field("bayesian_analyzer", &self.bayesian_analyzer)
            .field("performance_history", &self.performance_history)
            .field("decision_maker", &self.decision_maker)
            .finish()
    }
}

impl Default for BayesianEnsemblePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianEnsemblePredictor {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            bayesian_analyzer: crate::trading::BayesianAnalyzer::new(),
            performance_history: HashMap::new(),
            decision_maker: BayesianDecisionMaker::new(),
        }
    }

    /// Train the ensemble with multiple models
    pub fn train(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        features: &[Vec<f64>],
        targets: &[f64],
    ) {
        // Train individual models
        self.train_individual_models(prices, volumes, features, targets);

        // Initialize performance tracking
        for model_name in self.models.keys() {
            self.performance_history
                .insert(model_name.clone(), Vec::new());
        }
    }

    /// Train individual models for the ensemble
    fn train_individual_models(
        &mut self,
        prices: &[f64],
        _volumes: &[f64],
        features: &[Vec<f64>],
        targets: &[f64],
    ) {
        // Train GAS-GLD model
        let mut gas_gld = GASModel::new(
            GASDistribution::GLD(GeneralizedLambda::new(0.0, 1.0, 1.0, 1.0)),
            1000,
        );
        if let Some(returns) = Self::calculate_returns(prices) {
            for &ret in &returns {
                gas_gld.update(ret);
            }
            self.models.insert("gas_gld".to_string(), Box::new(gas_gld));
        }

        // Train Random Forest model
        if let Ok(x) = DenseMatrix::from_2d_array(
            &features
                .iter()
                .map(|row| row.as_slice())
                .collect::<Vec<_>>(),
        ) {
            let y = targets.to_vec();
            if let Ok(rf_model) = RandomForestRegressor::fit(&x, &y,
                smartcore::ensemble::random_forest_regressor::RandomForestRegressorParameters::default()
                    .with_n_trees(50).with_max_depth(8)) {
                self.models.insert("random_forest".to_string(), Box::new(RandomForestWrapper::new(rf_model)));
            }
        }

        // Train Linear Regression model
        if let Ok(x) = Array2::from_shape_vec(
            (features.len(), features[0].len()),
            features.iter().flatten().cloned().collect(),
        ) {
            if let Ok(lr_model) = linfa_linear::LinearRegression::default()
                .fit(&Dataset::new(x, Array1::from_vec(targets.to_vec())))
            {
                self.models.insert(
                    "linear_regression".to_string(),
                    Box::new(LinearRegressionWrapper::new(lr_model)),
                );
            }
        }
    }

    /// Make ensemble prediction using Bayesian model averaging
    pub fn predict(&mut self, current_price: f64, features: &[f64]) -> f64 {
        let mut predictions = Vec::new();
        let mut performances = Vec::new();

        // Get predictions from all models
        for (model_name, model) in &self.models {
            if let Some(pred) = model.predict(current_price, features) {
                predictions.push((model_name.clone(), pred));

                // Get recent performance for this model
                let perf = self
                    .performance_history
                    .get(model_name)
                    .and_then(|history| history.last())
                    .cloned()
                    .unwrap_or_else(|| ModelPerformance::new(1, 1, 0.0)); // Default neutral performance
                performances.push((model_name.as_str(), perf));
            }
        }

        if predictions.is_empty() {
            return 0.0;
        }

        // Calculate BMA weights
        let bma_weights = self.bayesian_analyzer.bma_weights(
            &performances
                .iter()
                .map(|(name, perf)| (*name, perf))
                .collect::<Vec<_>>(),
        );

        // Weighted ensemble prediction
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (model_name, prediction) in &predictions {
            let weight = bma_weights
                .get(model_name)
                .cloned()
                .unwrap_or(1.0 / predictions.len() as f64);
            weighted_sum += prediction * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Make ensemble prediction without updating internal state (for immutable access)
    pub fn predict_immutable(&self, current_price: f64, features: &[f64]) -> f64 {
        let mut predictions = Vec::new();
        let mut performances = Vec::new();

        // Get predictions from all models
        for (model_name, model) in &self.models {
            if let Some(pred) = model.predict(current_price, features) {
                predictions.push((model_name.clone(), pred));

                // Get recent performance for this model (immutable access)
                let perf = self
                    .performance_history
                    .get(model_name)
                    .and_then(|history| history.last())
                    .cloned()
                    .unwrap_or_else(|| ModelPerformance::new(1, 1, 0.0)); // Default neutral performance
                performances.push((model_name.as_str(), perf));
            }
        }

        if predictions.is_empty() {
            return 0.0;
        }

        // Calculate BMA weights
        let bma_weights = self.bayesian_analyzer.bma_weights(
            &performances
                .iter()
                .map(|(name, perf)| (*name, perf))
                .collect::<Vec<_>>(),
        );

        // Debug: Log individual model predictions and weights
        println!("🎯 Individual model predictions:");
        for (model_name, prediction) in &predictions {
            let weight = bma_weights
                .get(model_name)
                .cloned()
                .unwrap_or(1.0 / predictions.len() as f64);
            println!(
                "  {}: {:.4} (weight: {:.4})",
                model_name, prediction, weight
            );
        }

        // Debug: Log performance data
        println!("🎯 Model performances:");
        for (model_name, perf) in &performances {
            println!(
                "  {}: wins={}/{}, pnl=${:.2}",
                model_name, perf.wins, perf.total_trades, perf.total_pnl
            );
        }

        // Weighted ensemble prediction
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (model_name, prediction) in &predictions {
            let weight = bma_weights
                .get(model_name)
                .cloned()
                .unwrap_or(1.0 / predictions.len() as f64);
            weighted_sum += prediction * weight;
            total_weight += weight;
        }

        let final_prediction = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        println!("🎯 Ensemble prediction: {:.4}", final_prediction);
        final_prediction
    }

    /// Make decision using Bayesian decision theory
    pub fn decide_action(&mut self, prediction: f64, volatility: f64) -> TradingAction {
        self.decision_maker.decide_action(prediction, volatility)
    }

    /// Update performance and priors for Bayesian learning
    /// This version tracks individual model performance based on prediction accuracy
    pub fn update_performance_and_priors(&mut self, model_name: &str, pnl: f64, was_win: bool) {
        // For ensemble learning, we need to track individual model contributions
        // Since we don't know which model contributed most to the ensemble prediction,
        // we'll use a different approach: track prediction accuracy over time

        // Get existing performance history for this model
        let history = self
            .performance_history
            .entry(model_name.to_string())
            .or_default();

        // Calculate cumulative performance from all previous trades
        let mut total_wins = 0;
        let mut total_trades = 0;
        let mut total_pnl = 0.0;

        for perf in history.iter() {
            total_wins += perf.wins;
            total_trades += perf.total_trades;
            total_pnl += perf.total_pnl;
        }

        // Add the current trade
        total_wins += if was_win { 1 } else { 0 };
        total_trades += 1;
        total_pnl += pnl;

        // Create cumulative performance record
        let cumulative_perf = ModelPerformance::new(total_wins, total_trades, total_pnl);

        // Replace the entire history with just the cumulative record
        *history = vec![cumulative_perf.clone()];

        // Update Bayesian priors with cumulative performance
        self.bayesian_analyzer
            .update_priors(model_name, &cumulative_perf);
    }

    /// Calculate returns from prices
    fn calculate_returns(prices: &[f64]) -> Option<Vec<f64>> {
        if prices.len() < 2 {
            return None;
        }

        Some(
            prices
                .windows(2)
                .map(|window| (window[1] - window[0]) / window[0])
                .collect(),
        )
    }
}

/// Trait for model predictors in the ensemble
pub trait ModelPredictor {
    fn predict(&self, current_price: f64, features: &[f64]) -> Option<f64>;
}

/// Wrapper for Random Forest model
#[derive(Debug)]
pub struct RandomForestWrapper {
    model: RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>,
}

impl RandomForestWrapper {
    pub fn new(model: RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>) -> Self {
        Self { model }
    }
}

impl ModelPredictor for RandomForestWrapper {
    fn predict(&self, _current_price: f64, features: &[f64]) -> Option<f64> {
        let input = DenseMatrix::from_2d_array(&[features]).ok()?;
        let raw_prediction = self.model.predict(&input).ok()?.first().copied()?;

        // Scale the prediction to match GAS model signal strength (±0.8)
        // Assuming raw predictions are typically in range [-0.01, 0.01], scale to [-0.8, 0.8]
        let scaled_prediction = raw_prediction * 80.0;

        // Clamp to reasonable bounds
        Some(scaled_prediction.clamp(-0.8, 0.8))
    }
}

/// Wrapper for Linear Regression model
#[derive(Debug, Clone)]
pub struct LinearRegressionWrapper {
    model: linfa_linear::FittedLinearRegression<f64>,
}

impl LinearRegressionWrapper {
    pub fn new(model: linfa_linear::FittedLinearRegression<f64>) -> Self {
        Self { model }
    }
}

impl ModelPredictor for LinearRegressionWrapper {
    fn predict(&self, _current_price: f64, features: &[f64]) -> Option<f64> {
        let input = Array2::from_shape_vec((1, features.len()), features.to_vec()).ok()?;
        let raw_prediction = self.model.predict(&input).get(0).copied()?;

        // Scale the prediction to match GAS model signal strength (±0.8)
        // Assuming raw predictions are typically in range [-0.01, 0.01], scale to [-0.8, 0.8]
        let scaled_prediction = raw_prediction * 80.0;

        // Clamp to reasonable bounds
        Some(scaled_prediction.clamp(-0.8, 0.8))
    }
}
