use crate::ml::SimpleMLPredictor;
use crate::types::{TradingConfig, TradingPair};
use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use tracing::info;

#[derive(Debug)]
pub struct PositionSizer {
    pub base_risk_pct: f64, // Base risk per trade (e.g., 0.02 = 2%)
    pub max_risk_pct: f64,  // Maximum risk per trade (e.g., 0.05 = 5%)
    pub volatility_window: usize,
    pub recent_trades: VecDeque<bool>, // true = win, false = loss
    pub performance_window: usize,
    pub kelly_fraction: f64, // Kelly criterion multiplier (0.5 = half Kelly)
}

impl Default for PositionSizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionSizer {
    pub fn new() -> Self {
        Self {
                base_risk_pct: 1.0,  // Extremely aggressive: 100% base risk - risk entire balance per trade
            max_risk_pct: 1.0,  // Very high: 100% max risk
            volatility_window: 20,
            recent_trades: VecDeque::with_capacity(20),
            performance_window: 20,
            kelly_fraction: 0.8, // More aggressive Kelly (was 0.5)
        }
    }

    pub fn calculate_position_size(&self, _balance: f64, _price: f64, _volatility: f64, _confidence: f64) -> f64 {
        // Return the full base risk percentage without any adjustments
        // This allows for maximum capital utilization per trade
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
        Self {
            predictor: SimpleMLPredictor::new(50),
            position_sizer: PositionSizer::new(),
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
            
            // More aggressive trading: lower thresholds, especially for high-performing pairs
            let (buy_threshold, sell_threshold) = match self.pair {
                TradingPair::XRP | TradingPair::ETH => (0.1, -0.1), // Very aggressive for best performers
                _ => (0.3, -0.3), // Still aggressive but less so for others
            };
            
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
        
        let risk_pct = self.position_sizer.calculate_position_size(self.balance, price, volatility, confidence);
        
        // Use the full risk percentage without any adjustments for maximum capital utilization
        let adjusted_risk_pct = risk_pct.min(self.max_position_size_pct);
        
        let position_value = self.balance * adjusted_risk_pct;
        let position_size = position_value / price;

        self.position = position_size;
        self.entry_price = Some(price);
        self.balance -= position_size * price;
        self.total_trades += 1;

        info!("BUY: {:.6} units at ${}, position: {:.6}, remaining balance: ${:.2}, size: {:.1}% (vol: {:.3}, conf: {:.2})",
              position_size, price, self.position, self.balance, adjusted_risk_pct * 100.0, volume, confidence);
    }

    pub fn sell(&mut self, price: f64) {
        if self.position > 0.0 {
            let pnl = (price - self.entry_price.unwrap()) * self.position;
            self.balance += self.position * price;
            self.total_pnl += pnl;

            if pnl > 0.0 {
                self.winning_trades += 1;
            }

            info!("SELL: {:.6} units at ${}, P&L: ${:.2}, Total P&L: ${:.2}",
                  self.position, price, pnl, self.total_pnl);

            self.position = 0.0;
            self.entry_price = None;
        }
    }

    pub fn detect_trend_strength(&self) -> f64 {
        // Use multiple timeframes to detect trend strength
        let short_trend = self.predictor.calculate_momentum_from_index(
            self.predictor.trades.len().saturating_sub(1), 5
        ).unwrap_or(0.0);
        
        let medium_trend = self.predictor.calculate_momentum_from_index(
            self.predictor.trades.len().saturating_sub(1), 20
        ).unwrap_or(0.0);
        
        // Combine short and medium term trends
        // Weight recent trend more heavily
        (short_trend * 0.7 + medium_trend * 0.3).clamp(-1.0, 1.0)
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
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new()
    }
}

impl Backtester {
    pub fn new() -> Self {
        let config = TradingConfig::default();
        let mut traders = HashMap::new();

        for pair in &config.pairs {
            traders.insert(pair.clone(), PairTrader::new(&config, pair.clone()));
        }

        Self { traders, config }
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

        for (pair, trader) in &self.traders {
            trader.print_results(pair);
            total_pnl += trader.total_pnl;
            total_trades += trader.total_trades;
            total_wins += trader.winning_trades;
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

            // Check for stop loss / take profit
            if let Some(entry_price) = trader.entry_price {
                if trader.position > 0.0 {
                    let pnl_pct = (price - entry_price) / entry_price;
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