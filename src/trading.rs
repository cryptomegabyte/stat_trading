use crate::ml::SimpleMLPredictor;
use crate::types::{TradingConfig, TradingPair};
use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use tracing::info;
use std::collections::HashMap as HashMap2;
use statrs::distribution::{Beta, Normal, ContinuousCDF};
use statrs::function::beta::beta;

/// Bayesian analysis utilities for trading model comparison and uncertainty quantification
#[derive(Debug, Clone)]
pub struct BayesianAnalyzer {
    /// Prior beliefs about model performance (beta distribution parameters)
    model_priors: HashMap2<String, (f64, f64)>, // (alpha, beta) for beta distribution
}

impl Default for BayesianAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianAnalyzer {
    pub fn new() -> Self {
        let mut model_priors = HashMap2::new();

        // Default priors: Beta(2, 2) - slightly optimistic about model performance
        model_priors.insert("random_forest".to_string(), (2.0, 2.0));
        model_priors.insert("gas_ghd".to_string(), (2.0, 2.0));
        model_priors.insert("gas_vg".to_string(), (2.0, 2.0));
        model_priors.insert("gas_nig".to_string(), (2.0, 2.0));
        model_priors.insert("gas_gld".to_string(), (2.0, 2.0));
        model_priors.insert("hybrid_egarch_lstm".to_string(), (2.0, 2.0));

        Self { model_priors }
    }

    /// Calculate Bayes factor comparing two models
    pub fn bayes_factor(&self, model1: &str, model2: &str, data1: &ModelPerformance, data2: &ModelPerformance) -> f64 {
        let prior1 = self.model_priors.get(model1).unwrap_or(&(2.0, 2.0));
        let prior2 = self.model_priors.get(model2).unwrap_or(&(2.0, 2.0));

        let marg_like1 = self.beta_binomial_marginal_likelihood(data1.wins, data1.total_trades, prior1.0, prior1.1);
        let marg_like2 = self.beta_binomial_marginal_likelihood(data2.wins, data2.total_trades, prior2.0, prior2.1);

        marg_like1 / marg_like2
    }

    /// Compute marginal likelihood for beta-binomial model
    fn beta_binomial_marginal_likelihood(&self, wins: usize, total: usize, alpha: f64, beta_param: f64) -> f64 {
        let losses = total - wins;
        let beta_func_prior = beta(alpha, beta_param);
        let beta_func_post = beta(alpha + wins as f64, beta_param + losses as f64);
        beta_func_post / beta_func_prior
    }

    /// Calculate posterior probability that model1 is better than model2
    pub fn posterior_model_probability(&self, model1: &str, model2: &str, data1: &ModelPerformance, data2: &ModelPerformance) -> f64 {
        let bf = self.bayes_factor(model1, model2, data1, data2);
        bf / (bf + 1.0)
    }

    /// Calculate credible interval for win rate
    pub fn win_rate_credible_interval(&self, model: &str, data: &ModelPerformance, credibility: f64) -> (f64, f64) {
        let prior = self.model_priors.get(model).unwrap_or(&(2.0, 2.0));
        let post_alpha = prior.0 + data.wins as f64;
        let post_beta = prior.1 + (data.total_trades - data.wins) as f64;
        let beta_dist = Beta::new(post_alpha, post_beta).unwrap();

        let lower_quantile = (1.0 - credibility) / 2.0;
        let upper_quantile = 1.0 - lower_quantile;

        (
            beta_dist.inverse_cdf(lower_quantile),
            beta_dist.inverse_cdf(upper_quantile)
        )
    }

    /// Calculate expected P&L with uncertainty
    pub fn pnl_credible_interval(&self, data: &ModelPerformance, credibility: f64) -> (f64, f64) {
        if data.total_trades == 0 {
            return (0.0, 0.0);
        }

        let mean_pnl = data.total_pnl / data.total_trades as f64;
        let variance = data.pnl_variance.unwrap_or(100.0);

        let std_dev = variance.sqrt() / (data.total_trades as f64).sqrt();

        let normal_dist = Normal::new(mean_pnl, std_dev).unwrap();

        let lower_quantile = (1.0 - credibility) / 2.0;
        let upper_quantile = 1.0 - lower_quantile;

        (
            normal_dist.inverse_cdf(lower_quantile),
            normal_dist.inverse_cdf(upper_quantile)
        )
    }

    /// Bayesian model averaging weights with fallback to fixed weights
    pub fn bma_weights(&self, models: &[(&str, &ModelPerformance)]) -> HashMap2<String, f64> {
        let mut weights = HashMap2::new();
        let mut total_weight = 0.0;

        // Check if all models have identical performance (common case in ensemble learning)
        let first_perf = models.first().map(|(_, perf)| perf);
        let all_identical = models.iter().all(|(_, perf)| {
            perf.wins == first_perf.unwrap().wins &&
            perf.total_trades == first_perf.unwrap().total_trades &&
            (perf.total_pnl - first_perf.unwrap().total_pnl).abs() < 0.01
        });

        if all_identical && models.len() > 1 {
            // When all models have identical performance (as in ensemble learning),
            // use fixed weights based on model reliability characteristics
            for (model_name, _) in models {
                let weight = match *model_name {
                    "random_forest" => 0.4,      // ML models get higher weight
                    "linear_regression" => 0.4,  // ML models get higher weight
                    "gas_gld" => 0.2,            // GAS models are more volatile, lower weight
                    _ => 1.0 / models.len() as f64, // Default equal weight
                };
                weights.insert(model_name.to_string(), weight);
                total_weight += weight;
            }
        } else {
            // Use traditional BMA when models have different performance
            for (model_name, performance) in models {
                let prior = self.model_priors.get(*model_name).unwrap_or(&(2.0, 2.0));
                let marg_like = self.beta_binomial_marginal_likelihood(
                    performance.wins,
                    performance.total_trades,
                    prior.0,
                    prior.1
                );
                weights.insert(model_name.to_string(), marg_like);
                total_weight += marg_like;
            }
        }

        // Normalize weights
        for weight in weights.values_mut() {
            *weight /= total_weight;
        }

        weights
    }

    /// Update model priors based on observed performance data
    /// This implements Bayesian parameter updating for model beliefs
    pub fn update_priors(&mut self, model_name: &str, performance: &ModelPerformance) {
        let current_prior = self.model_priors.get(model_name).unwrap_or(&(2.0, 2.0));

        // Update priors using empirical Bayes approach
        // New prior = current_prior + observed_data
        let updated_alpha = current_prior.0 + performance.wins as f64;
        let updated_beta = current_prior.1 + (performance.total_trades - performance.wins) as f64;

        // Apply some regularization to prevent over-confidence
        let regularization = 0.1;
        let final_alpha = (updated_alpha * (1.0 - regularization)) + (current_prior.0 * regularization);
        let final_beta = (updated_beta * (1.0 - regularization)) + (current_prior.1 * regularization);

        self.model_priors.insert(model_name.to_string(), (final_alpha, final_beta));
    }

    /// Update priors for multiple models at once
    pub fn update_priors_batch(&mut self, performances: &[(&str, &ModelPerformance)]) {
        for (model_name, performance) in performances {
            self.update_priors(model_name, performance);
        }
    }

    /// Get current prior beliefs for analysis
    pub fn get_priors(&self) -> &HashMap2<String, (f64, f64)> {
        &self.model_priors
    }

    /// Reset priors to default values (useful for testing)
    pub fn reset_priors(&mut self) {
        self.model_priors.clear();
        let default_priors = [
            ("random_forest", (2.0, 2.0)),
            ("gas_ghd", (2.0, 2.0)),
            ("gas_vg", (2.0, 2.0)),
            ("gas_nig", (2.0, 2.0)),
            ("gas_gld", (2.0, 2.0)),
            ("hybrid_egarch_lstm", (2.0, 2.0)),
        ];

        for (model, prior) in default_priors {
            self.model_priors.insert(model.to_string(), prior);
        }
    }
}

/// Performance data for a trading model
#[derive(Debug, Clone)]
pub struct ModelPerformance {
    pub wins: usize,
    pub total_trades: usize,
    pub total_pnl: f64,
    pub pnl_variance: Option<f64>,
}

impl ModelPerformance {
    pub fn new(wins: usize, total_trades: usize, total_pnl: f64) -> Self {
        Self {
            wins,
            total_trades,
            total_pnl,
            pnl_variance: None,
        }
    }

    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            self.wins as f64 / self.total_trades as f64
        }
    }

    pub fn average_pnl(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            self.total_pnl / self.total_trades as f64
        }
    }
}

#[derive(Debug)]
pub struct PositionSizer {
    pub base_risk_pct: f64, // Base risk per trade (e.g., 0.02 = 2%)
    pub max_risk_pct: f64,  // Maximum risk per trade (e.g., 0.05 = 5%)
    pub volatility_window: usize,
    pub recent_trades: VecDeque<bool>, // true = win, false = loss
    pub performance_window: usize,
    pub kelly_fraction: f64, // Kelly criterion multiplier (0.5 = half Kelly)
    pub leverage: f64, // Leverage multiplier (e.g., 2.0 = 2x leverage)
}

impl Default for PositionSizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionSizer {
    pub fn new() -> Self {
        Self::with_leverage(1.0) // Spot trading: no leverage
    }

    pub fn with_leverage(leverage: f64) -> Self {
        Self {
                base_risk_pct: 0.5,  // EXTREME: 50% base risk for aggressive spot trading
            max_risk_pct: 0.5,  // Very high: 50% max risk
            volatility_window: 20,
            recent_trades: VecDeque::with_capacity(20),
            performance_window: 20,
            kelly_fraction: 1.0, // Full Kelly criterion for maximum aggression
            leverage,
        }
    }

    pub fn calculate_position_size(&self, _balance: f64, _price: f64, _volatility: f64, _confidence: f64) -> f64 {
        // For spot trading: return the base risk percentage (no leverage)
        // This gives the actual percentage of balance to risk per trade
        self.base_risk_pct
    }

    pub fn record_trade_result(&mut self, was_win: bool) {
        self.recent_trades.push_back(was_win);
        if self.recent_trades.len() > self.performance_window {
            self.recent_trades.pop_front();
        }
    }

    pub fn get_current_volatility(&self, predictor: &SimpleMLPredictor) -> f64 {
        predictor.calculate_volatility(self.volatility_window).unwrap_or(0.01)
    }

    pub fn get_signal_confidence(&self, predictor: &SimpleMLPredictor) -> f64 {
        // Simple confidence based on how far prediction is from current price
        if let (Some(current), Some(predicted)) = (predictor.trades.back(), predictor.predict_next()) {
            let diff_pct = ((predicted - current.price) / current.price).abs();
            // Higher confidence for larger predicted moves
            (diff_pct * 10.0).clamp(0.1, 1.0)
        } else {
            0.5 // Default confidence
        }
    }
}

#[derive(Debug)]
pub struct PairTrader {
    pub predictor: SimpleMLPredictor,
    pub position_sizer: PositionSizer,
    pub balance: f64,
    pub position: f64, // amount of crypto held
    pub entry_price: Option<f64>,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub total_pnl: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub max_position_size_pct: f64,
    pub pair: TradingPair, // Add pair field
}

impl PairTrader {
    pub fn new(config: &TradingConfig, pair: TradingPair) -> Self {
        Self::new_with_model(config, pair, "hybrid_egarch_lstm")
    }

    pub fn new_with_model(config: &TradingConfig, pair: TradingPair, model_type: &str) -> Self {
        let mut predictor = SimpleMLPredictor::new(50);
        // Set the model type for training
        predictor.model_type = model_type.to_string();
        
        Self {
            predictor,
            position_sizer: PositionSizer::with_leverage(config.leverage),
            balance: config.initial_balance, // Use the allocated balance per pair
            position: 0.0,
            entry_price: None,
            total_trades: 0,
            winning_trades: 0,
            total_pnl: 0.0,
            stop_loss_pct: config.stop_loss_pct,
            take_profit_pct: config.take_profit_pct,
            max_position_size_pct: config.max_position_size_pct,
            pair, // Store the pair
        }
    }

    pub fn process_trade(&mut self, price: f64, volume: f64) {
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

        // Generate trading signal with trend confirmation
        if let Some(signal) = self.predictor.predict_next() {
            let trend_strength = self.detect_trend_strength();
            
            // Dynamic thresholds based on ensemble prediction volatility and performance
            let (buy_threshold, sell_threshold) = self.calculate_dynamic_thresholds(signal);
            
            if signal > 0.0 && self.can_buy(price) {
                if trend_strength > -0.2 || signal > buy_threshold {
                    self.buy(price, volume);
                }
            } else if signal < 0.0 && self.can_sell()
                && (trend_strength < 0.2 || signal < sell_threshold) {
                    self.sell(price);
                }
        }
    }

    pub fn can_buy(&self, price: f64) -> bool {
        self.position == 0.0 && self.balance >= price * 0.001 // minimum 0.001 units
    }

    pub fn can_sell(&self) -> bool {
        self.position > 0.0
    }

    pub fn buy(&mut self, price: f64, volume: f64) {
        let volatility = self.position_sizer.get_current_volatility(&self.predictor);
        let confidence = self.position_sizer.get_signal_confidence(&self.predictor);

        // Use VaR-based position sizing for better risk management
        let var_position_size = self.predictor.get_var_position_size(self.balance, 0.02); // 2% max loss
        let traditional_risk_pct = self.position_sizer.calculate_position_size(self.balance, price, volatility, confidence);

        // Use the more conservative of the two approaches
        let risk_pct = var_position_size.min(traditional_risk_pct);

        // Use the risk percentage, capped at max position size
        let adjusted_risk_pct = risk_pct.min(self.max_position_size_pct);
        let position_value = self.balance * adjusted_risk_pct;
        let position_size = position_value / price;

        self.position = position_size;
        self.entry_price = Some(price);
        // In spot trading, deduct the full position value from balance
        self.balance -= position_size * price;
        self.total_trades += 1;

        info!("BUY: {:.6} units at ${}, position: {:.6}, remaining balance: ${:.2}, size: {:.1}% (VaR-adjusted, vol: {:.3}, conf: {:.2})",
              position_size, price, self.position, self.balance, adjusted_risk_pct * 100.0, volume, confidence);
    }

    pub fn sell(&mut self, price: f64) {
        if self.position > 0.0 {
            let pnl = (price - self.entry_price.unwrap()) * self.position;
            self.balance += self.position * price;
            self.total_pnl += pnl;

            let was_win = pnl > 0.0;
            if was_win {
                self.winning_trades += 1;
            }

            // Update ensemble model performance for Bayesian learning
            self.predictor.update_ensemble_performance(pnl, was_win);

            info!("SELL: {:.6} units at ${}, P&L: ${:.2}, Total P&L: ${:.2}",
                  self.position, price, pnl, self.total_pnl);

            self.position = 0.0;
            self.entry_price = None;
        }
    }

    pub fn detect_trend_strength(&self) -> f64 {
        if self.predictor.trades.len() < 20 {
            return 0.0; // Not enough data for trend analysis
        }

        let current_idx = self.predictor.trades.len().saturating_sub(1);

        // Multiple timeframe momentum analysis
        let short_momentum = self.predictor.calculate_momentum_from_index(current_idx, 5).unwrap_or(0.0);
        let medium_momentum = self.predictor.calculate_momentum_from_index(current_idx, 10).unwrap_or(0.0);
        let long_momentum = self.predictor.calculate_momentum_from_index(current_idx, 20).unwrap_or(0.0);

        // Moving average trend analysis
        let sma5 = self.predictor.calculate_sma_from_index(current_idx, 5).unwrap_or(0.0);
        let sma10 = self.predictor.calculate_sma_from_index(current_idx, 10).unwrap_or(0.0);
        let sma20 = self.predictor.calculate_sma_from_index(current_idx, 20).unwrap_or(0.0);

        // Calculate MA slope (trend direction and strength)
        let ma_slope_short = if sma5 > 0.0 && sma10 > 0.0 {
            (sma5 - sma10) / sma10 // Percentage change
        } else { 0.0 };

        let ma_slope_medium = if sma10 > 0.0 && sma20 > 0.0 {
            (sma10 - sma20) / sma20 // Percentage change
        } else { 0.0 };

        // RSI for overbought/oversold confirmation
        let rsi = self.predictor.calculate_rsi_from_index(current_idx, 14).unwrap_or(50.0);
        let rsi_signal = if rsi > 70.0 {
            -0.3 // Overbought - bearish
        } else if rsi < 30.0 {
            0.3 // Oversold - bullish
        } else {
            0.0 // Neutral
        };

        // MACD for trend confirmation
        let macd = self.predictor.calculate_macd_from_index(current_idx).unwrap_or(0.0);
        let macd_signal = macd.signum() * macd.abs().min(0.5); // Normalize and cap

        // Volume trend analysis
        let volume_trend = self.predictor.calculate_volume_trend_from_index(current_idx, 10).unwrap_or(0.0);

        // Combine all signals with weights
        let momentum_score = (short_momentum * 0.4 + medium_momentum * 0.4 + long_momentum * 0.2).clamp(-1.0, 1.0);
        let ma_score = (ma_slope_short * 0.6 + ma_slope_medium * 0.4).clamp(-1.0, 1.0);
        let technical_score = (rsi_signal * 0.3 + macd_signal * 0.4 + volume_trend * 0.3).clamp(-1.0, 1.0);

        // Final trend strength: weighted combination
        let trend_strength = (momentum_score * 0.4 + ma_score * 0.4 + technical_score * 0.2).clamp(-1.0, 1.0);

        info!("📈 Trend analysis - momentum: {:.3}, MA: {:.3}, technical: {:.3}, final: {:.3}",
              momentum_score, ma_score, technical_score, trend_strength);

        trend_strength
    }

    /// Calculate dynamic thresholds based on ensemble prediction volatility and performance
    pub fn calculate_dynamic_thresholds(&self, current_signal: f64) -> (f64, f64) {
        // Base thresholds that adapt to the ensemble's signal strength
        let base_buy_threshold: f64;
        let base_sell_threshold: f64;

        // Calculate recent prediction volatility from the predictor
        let prediction_volatility = self.predictor.get_prediction_volatility().unwrap_or(0.1);

        // Calculate win rate for dynamic adjustment
        let win_rate = if self.total_trades > 0 {
            self.winning_trades as f64 / self.total_trades as f64
        } else {
            0.5 // Default neutral win rate
        };

        // Adaptive base thresholds based on pair performance and volatility
        match self.pair {
            TradingPair::XRP | TradingPair::ETH => {
                // More aggressive for historically better performers
                base_buy_threshold = 0.05 + (prediction_volatility * 0.5);
                base_sell_threshold = -0.05 - (prediction_volatility * 0.5);
            },
            _ => {
                // Conservative for other pairs
                base_buy_threshold = 0.08 + (prediction_volatility * 0.7);
                base_sell_threshold = -0.08 - (prediction_volatility * 0.7);
            }
        }

        // Adjust thresholds based on recent performance
        // If winning, be more aggressive; if losing, be more conservative
        let performance_multiplier = if win_rate > 0.6 {
            0.8 // More aggressive when winning
        } else if win_rate < 0.4 {
            1.3 // More conservative when losing
        } else {
            1.0 // Neutral
        };

        let buy_threshold = (base_buy_threshold * performance_multiplier).min(0.15); // Cap at 15%
        let sell_threshold = (base_sell_threshold * performance_multiplier).max(-0.15); // Cap at -15%

        // Ensure thresholds are reasonable and don't cross zero
        let final_buy_threshold = buy_threshold.max(0.01);
        let final_sell_threshold = sell_threshold.min(-0.01);

        info!("🎯 Dynamic thresholds - Buy: {:.3}, Sell: {:.3} (vol: {:.3}, win_rate: {:.1}%, perf_mult: {:.2})",
              final_buy_threshold, final_sell_threshold, prediction_volatility, win_rate * 100.0, performance_multiplier);

        (final_buy_threshold, final_sell_threshold)
    }

    pub fn print_results(&self, pair: &TradingPair) {
        let win_rate = if self.total_trades > 0 {
            (self.winning_trades as f64 / self.total_trades as f64) * 100.0
        } else {
            0.0
        };

        info!("📊 {} Results:", pair.symbol().to_uppercase());
        info!("Total Trades: {}", self.total_trades);
        info!("Winning Trades: {}", self.winning_trades);
        info!("Win Rate: {:.2}%", win_rate);
        info!("Total P&L: ${:.2}", self.total_pnl);
        info!("Final Balance: ${:.2}", self.balance);
    }
}

#[derive(Debug)]
pub struct Backtester {
    pub traders: HashMap<TradingPair, PairTrader>,
    pub config: TradingConfig,
    pub bayesian_analyzer: BayesianAnalyzer,
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new()
    }
}

impl Backtester {
    pub fn new() -> Self {
        Self::with_model_type("hybrid_egarch_lstm")
    }

    pub fn with_model_type(model_type: &str) -> Self {
        let config = TradingConfig::default();
        let mut traders = HashMap::new();

        for pair in &config.pairs {
            traders.insert(pair.clone(), PairTrader::new_with_model(&config, pair.clone(), model_type));
        }

        Self {
            traders,
            config,
            bayesian_analyzer: BayesianAnalyzer::new(),
        }
    }

    pub fn process_trade(&mut self, pair: &TradingPair, price: f64, volume: f64) {
        if let Some(trader) = self.traders.get_mut(pair) {
            trader.process_trade(price, volume);
        }
    }

    pub fn print_results(&self) {
        info!("📊 Multi-Pair Backtest Results:");
        let mut total_pnl = 0.0;
        let mut total_trades = 0;
        let mut total_wins = 0;
        let mut model_performances = Vec::new();

        for (pair, trader) in &self.traders {
            trader.print_results(pair);
            total_pnl += trader.total_pnl;
            total_trades += trader.total_trades;
            total_wins += trader.winning_trades;

            // Collect performance data for Bayesian analysis
            let perf = ModelPerformance::new(
                trader.winning_trades,
                trader.total_trades,
                trader.total_pnl
            );
            model_performances.push((trader.predictor.model_type.clone(), perf));
        }

        let overall_win_rate = if total_trades > 0 {
            (total_wins as f64 / total_trades as f64) * 100.0
        } else {
            0.0
        };

        info!("🎯 Overall Results:");
        info!("Total Trades: {}", total_trades);
        info!("Winning Trades: {}", total_wins);
        info!("Win Rate: {:.2}%", overall_win_rate);
        info!("Total P&L: ${:.2}", total_pnl);
        info!("Final Balance: ${:.2}", 2000.0 + total_pnl);

        // Bayesian Analysis
        self.print_bayesian_analysis(&model_performances);
    }

    fn print_bayesian_analysis(&self, performances: &[(String, ModelPerformance)]) {
        if performances.is_empty() {
            return;
        }

        info!("\n🧠 Bayesian Model Analysis:");

        // Calculate BMA weights
        let model_refs: Vec<(&str, &ModelPerformance)> = performances.iter()
            .map(|(name, perf)| (name.as_str(), perf))
            .collect();
        let bma_weights = self.bayesian_analyzer.bma_weights(&model_refs);
        info!("📈 Bayesian Model Averaging Weights:");
        for (model, weight) in &bma_weights {
            info!("  {}: {:.3} ({:.1}%)", model, weight, weight * 100.0);
        }

        // Compare models using Bayes factors
        if performances.len() >= 2 {
            info!("\n⚖️  Bayesian Model Comparison (Bayes Factors):");
            for i in 0..performances.len() {
                for j in (i+1)..performances.len() {
                    let (model1, perf1) = &performances[i];
                    let (model2, perf2) = &performances[j];

                    let bf = self.bayesian_analyzer.bayes_factor(model1, model2, perf1, perf2);
                    let prob1_better = self.bayesian_analyzer.posterior_model_probability(model1, model2, perf1, perf2);

                    let comparison = if bf > 1.0 {
                        format!("{} favored {:.1}x", model1, bf)
                    } else {
                        format!("{} favored {:.1}x", model2, 1.0/bf)
                    };

                    info!("  {} vs {}: {} (P({} better) = {:.1}%)",
                          model1, model2, comparison, model1, prob1_better * 100.0);
                }
            }
        }

        // Credible intervals for win rates
        info!("\n📊 Win Rate Credible Intervals (95%):");
        for (model, perf) in performances {
            let (lower, upper) = self.bayesian_analyzer.win_rate_credible_interval(model, perf, 0.95);
            info!("  {}: {:.1}% - {:.1}% (observed: {:.1}%)",
                  model, lower * 100.0, upper * 100.0, perf.win_rate() * 100.0);
        }

        // Credible intervals for P&L
        info!("\n💰 P&L Credible Intervals (95%):");
        for (model, perf) in performances {
            let (lower, upper) = self.bayesian_analyzer.pnl_credible_interval(perf, 0.95);
            info!("  {}: ${:.2} - ${:.2} (observed: ${:.2})",
                  model, lower, upper, perf.average_pnl());
        }
    }
}

#[derive(Debug)]
pub struct LiveTrader {
    pub traders: HashMap<TradingPair, PairTrader>,
    pub config: TradingConfig,
}

impl Default for LiveTrader {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveTrader {
    pub fn new() -> Self {
        let config = TradingConfig::default();
        let mut traders = HashMap::new();

        for pair in &config.pairs {
            traders.insert(pair.clone(), PairTrader::new(&config, pair.clone()));
        }

        Self { traders, config }
    }

    pub async fn process_price_update(&mut self, pair: &TradingPair, price: f64, volume: f64) -> Result<()> {
        if let Some(trader) = self.traders.get_mut(pair) {
            trader.predictor.add_trade(price, volume);

            // Check for stop loss / take profit / VaR-based risk management
            if let Some(entry_price) = trader.entry_price {
                if trader.position > 0.0 {
                    let pnl = (price - entry_price) * trader.position;
                    let pnl_pct = (price - entry_price) / entry_price;
                    
                    // Check VaR-based risk limits
                    if trader.predictor.should_close_based_on_var(pnl, trader.position * price) {
                        info!("🛡️ {} VaR RISK LIMIT triggered (P&L: ${:.2}, {:.2}%)", 
                              pair.symbol().to_uppercase(), pnl, pnl_pct * 100.0);
                        Self::execute_sell(trader, price).await?;
                        return Ok(());
                    }
                    
                    // Traditional stop loss
                    if pnl_pct <= -trader.stop_loss_pct {
                        info!("🛑 {} STOP LOSS triggered at {:.2}%", pair.symbol().to_uppercase(), pnl_pct * 100.0);
                        Self::execute_sell(trader, price).await?;
                        return Ok(());
                    } else if pnl_pct >= trader.take_profit_pct {
                        info!("🎯 {} TAKE PROFIT triggered at {:.2}%", pair.symbol().to_uppercase(), pnl_pct * 100.0);
                        Self::execute_sell(trader, price).await?;
                        return Ok(());
                    }
                }
            }

            // Get trading signal
            if let Some(signal) = trader.predictor.predict_next() {
                if signal > 0.0 && trader.can_buy(price) {
                    Self::execute_buy(trader, price, volume).await?;
                } else if signal < 0.0 && trader.can_sell() {
                    Self::execute_sell(trader, price).await?;
                }
            }
        }

        Ok(())
    }

    async fn execute_buy(trader: &mut PairTrader, price: f64, _volume: f64) -> Result<()> {
        let volatility = trader.position_sizer.get_current_volatility(&trader.predictor);
        let confidence = trader.position_sizer.get_signal_confidence(&trader.predictor);
        let position_size_pct = trader.position_sizer.calculate_position_size(trader.balance, price, volatility, confidence);
        let max_position_value = trader.balance * position_size_pct.min(trader.max_position_size_pct);
        let position_size = max_position_value / price;

        if position_size * price < 1.0 {
            info!("Trade too small (${:.2}), skipping", position_size * price);
            return Ok(());
        }

        info!("🚀 LIVE BUY: {:.6} units at ${:.2}, total: ${:.2}, size: {:.2}% (vol: {:.4}, conf: {:.2})",
              position_size, price, position_size * price, position_size_pct * 100.0, volatility, confidence);

        trader.position = position_size;
        trader.entry_price = Some(price);
        trader.balance -= position_size * price;
        trader.total_trades += 1;

        Ok(())
    }

    async fn execute_sell(trader: &mut PairTrader, price: f64) -> Result<()> {
        if trader.position <= 0.0 {
            return Ok(());
        }

        let units_to_sell = trader.position;
        let sale_value = units_to_sell * price;

        info!("💰 LIVE SELL: {:.6} units at ${:.2}, total: ${:.2}",
              units_to_sell, price, sale_value);

        trader.position = 0.0;
        trader.balance += sale_value;

        if let Some(entry) = trader.entry_price {
            let pnl = sale_value - (units_to_sell * entry);
            trader.total_pnl += pnl;
            if pnl > 0.0 {
                trader.winning_trades += 1;
            }
            info!("Trade P&L: ${:.2} ({:.2}%)", pnl, (pnl / (units_to_sell * entry)) * 100.0);
        }

        trader.entry_price = None;

        Ok(())
    }

    pub fn print_status(&self) {
        info!("📊 Multi-Pair Live Trading Status:");
        let mut total_balance = 0.0;
        let mut total_trades = 0;
        let mut total_wins = 0;
        let mut total_pnl = 0.0;

        for (pair, trader) in &self.traders {
            let win_rate = if trader.total_trades > 0 {
                trader.winning_trades as f64 / trader.total_trades as f64 * 100.0
            } else {
                0.0
            };

            info!("{}: Balance ${:.2}, Position {:.6}, Trades {}, Win Rate {:.1}%, P&L ${:.2}",
                  pair.symbol().to_uppercase(), trader.balance, trader.position, trader.total_trades, win_rate, trader.total_pnl);

            total_balance += trader.balance;
            total_trades += trader.total_trades;
            total_wins += trader.winning_trades;
            total_pnl += trader.total_pnl;
        }

        let overall_win_rate = if total_trades > 0 {
            total_wins as f64 / total_trades as f64 * 100.0
        } else {
            0.0
        };

        info!("🎯 Overall: Balance ${:.2}, Total Trades {}, Win Rate {:.1}%, Total P&L ${:.2}",
              total_balance, total_trades, overall_win_rate, total_pnl);
    }
}