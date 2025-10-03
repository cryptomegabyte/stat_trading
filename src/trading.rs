use crate::ml::SimpleMLPredictor;
use anyhow::Result;
use tracing::info;
use std::collections::VecDeque;

#[derive(Debug)]
pub struct PositionSizer {
    pub base_risk_pct: f64, // Base risk per trade (e.g., 0.02 = 2%)
    pub max_risk_pct: f64,  // Maximum risk per trade (e.g., 0.05 = 5%)
    pub volatility_window: usize,
    pub recent_trades: VecDeque<bool>, // true = win, false = loss
    pub performance_window: usize,
    pub kelly_fraction: f64, // Kelly criterion multiplier (0.5 = half Kelly)
}

impl PositionSizer {
    pub fn new() -> Self {
        Self {
                base_risk_pct: 0.05,  // 5% base risk for better position sizes
            max_risk_pct: 0.05,  // 5% max risk
            volatility_window: 20,
            recent_trades: VecDeque::with_capacity(20),
            performance_window: 20,
            kelly_fraction: 0.5, // Conservative Kelly
        }
    }

    pub fn calculate_position_size(&self, balance: f64, price: f64, volatility: f64, confidence: f64) -> f64 {
        // Base position size using risk percentage
        let base_risk_amount = balance * self.base_risk_pct;
        let mut position_value = base_risk_amount / price;

        // Adjust for volatility (higher volatility = smaller position)
        let volatility_adjustment = if volatility > 0.0 {
            let normalized_volatility = (volatility * 100.0).min(20.0); // Cap at 20%
            1.0 / (1.0 + normalized_volatility * 0.5) // Less aggressive reduction
        } else {
            1.0
        };

        // Adjust for recent performance using Kelly criterion approximation
        let performance_adjustment = self.calculate_performance_multiplier();

        // Adjust for signal confidence
        let confidence_adjustment = 0.5 + (confidence * 0.5); // 0.5 to 1.0 based on confidence

        // Apply all adjustments
        position_value *= volatility_adjustment;
        position_value *= performance_adjustment;
        position_value *= confidence_adjustment;

        // Cap at maximum risk
        let max_position_value = balance * self.max_risk_pct;
        let max_position_size = max_position_value / price;
        position_value = position_value.min(max_position_size);

        // Ensure minimum position size (0.001 BTC)
        position_value.max(0.001)
    }

    fn calculate_performance_multiplier(&self) -> f64 {
        if self.recent_trades.is_empty() {
            return 1.0; // No adjustment if no history
        }

        let wins = self.recent_trades.iter().filter(|&&win| win).count();
        let total = self.recent_trades.len();
        let win_rate = wins as f64 / total as f64;

        if win_rate > 0.6 {
            // Good performance - can risk a bit more
            1.2
        } else if win_rate < 0.4 {
            // Poor performance - reduce risk
            0.7
        } else {
            1.0
        }
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
            (diff_pct * 10.0).min(1.0).max(0.1)
        } else {
            0.5 // Default confidence
        }
    }
}

#[derive(Debug)]
pub struct Backtester {
    pub predictor: SimpleMLPredictor,
    pub position_sizer: PositionSizer,
    pub balance: f64,
    pub position: f64, // amount of BTC held
    pub entry_price: Option<f64>,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub total_pnl: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
}

impl Backtester {
    pub fn new() -> Self {
        Self {
            predictor: SimpleMLPredictor::new(50), // increased window size
            position_sizer: PositionSizer::new(),
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

        if let Some(signal) = self.predictor.get_trading_signal() {
            match signal.as_str() {
                "BUY" => self.buy(price),
                "SELL" => self.sell(price),
                _ => {}
            }
        }
    }

    pub fn buy(&mut self, price: f64) {
        if self.position == 0.0 && self.balance > 100.0 { // minimum balance
            let volatility = self.position_sizer.get_current_volatility(&self.predictor);
            let confidence = self.position_sizer.get_signal_confidence(&self.predictor);
            let position_size_pct = self.position_sizer.calculate_position_size(self.balance, price, volatility, confidence);
            let trade_amount = self.balance * position_size_pct;
            self.position = trade_amount / price;
            self.entry_price = Some(price);
            self.balance -= trade_amount;
            info!("BUY: {} BTC at ${}, position: {}, remaining balance: ${:.2}, size: {:.2}% (vol: {:.4}, conf: {:.2})",
                  self.position, price, self.position, self.balance, position_size_pct * 100.0, volatility, confidence);
        }
    }

    pub fn sell(&mut self, price: f64) {
        if self.position > 0.0 {
            let exit_value = self.position * price;
            let pnl = exit_value - (self.entry_price.unwrap() * self.position);
            self.total_pnl += pnl;
            self.balance += exit_value;
            self.total_trades += 1;
            if pnl > 0.0 {
                self.winning_trades += 1;
            }
            // Record trade result for position sizing
            let pnl_pct = pnl / (self.entry_price.unwrap() * self.position);
            self.position_sizer.record_trade_result(pnl_pct > 0.0);
            info!("SELL: {} BTC at ${}, P&L: ${:.2}, Total P&L: ${:.2}", self.position, price, pnl, self.total_pnl);
            self.position = 0.0;
            self.entry_price = None;
        }
    }

    pub fn print_results(&self) {
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
pub struct LiveTrader {
    pub predictor: SimpleMLPredictor,
    pub position_sizer: PositionSizer,
    pub balance: f64,
    pub position: f64, // amount of BTC held
    pub entry_price: Option<f64>,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub total_pnl: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub max_position_size_pct: f64, // max % of balance to use per trade
}

impl LiveTrader {
    pub fn new() -> Self {
        Self {
            predictor: SimpleMLPredictor::new(50),
            position_sizer: PositionSizer::new(),
            balance: 2000.0, // start with $2k for safety
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

    pub fn can_buy(&self, price: f64) -> bool {
        self.position == 0.0 && self.balance >= price * 0.001 // minimum 0.001 BTC
    }

    pub fn can_sell(&self) -> bool {
        self.position > 0.0
    }

    pub async fn execute_buy(&mut self, price: f64) -> Result<()> {
        let volatility = self.position_sizer.get_current_volatility(&self.predictor);
        let confidence = self.position_sizer.get_signal_confidence(&self.predictor);
        let position_size_pct = self.position_sizer.calculate_position_size(self.balance, price, volatility, confidence);
        let max_position_value = self.balance * position_size_pct.min(self.max_position_size_pct);
        let btc_to_buy = (max_position_value / price).min(self.balance / price);

        if btc_to_buy * price < 1.0 {
            info!("Trade too small (${:.2}), skipping", btc_to_buy * price);
            return Ok(());
        }

        // Here you would execute the actual buy order
        // For now, we'll simulate it
        info!("🚀 LIVE BUY: {:.6} BTC at ${:.2}, total: ${:.2}, size: {:.2}% (vol: {:.4}, conf: {:.2})",
              btc_to_buy, price, btc_to_buy * price, position_size_pct * 100.0, volatility, confidence);

        self.position += btc_to_buy;
        self.balance -= btc_to_buy * price;
        self.entry_price = Some(price);
        self.total_trades += 1;

        Ok(())
    }

    pub async fn execute_sell(&mut self, price: f64) -> Result<()> {
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
            // Record trade result for position sizing
            let pnl_pct = pnl / (btc_to_sell * entry);
            self.position_sizer.record_trade_result(pnl_pct > 0.0);
            info!("Trade P&L: ${:.2} ({:.2}%)", pnl, pnl_pct * 100.0);
        }

        self.entry_price = None;

        Ok(())
    }

    pub async fn process_price_update(&mut self, price: f64) -> Result<()> {
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
                "RANDOM" => {
                    // Special case for random trading when no ML model
                    if self.can_sell() {
                        // If we have a position, sell
                        self.execute_sell(price).await?;
                    } else if self.can_buy(price) {
                        // If we don't have a position, buy
                        self.execute_buy(price).await?;
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    pub fn print_status(&self) {
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