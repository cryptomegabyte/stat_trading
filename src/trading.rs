use crate::ml::SimpleMLPredictor;
use anyhow::Result;
use tracing::info;

#[derive(Debug)]
pub struct Backtester {
    pub predictor: SimpleMLPredictor,
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
            let trade_amount = self.balance * 0.1; // 10% of balance
            self.position = trade_amount / price;
            self.entry_price = Some(price);
            self.balance -= trade_amount;
            info!("BUY: {} BTC at ${}, position: {}, remaining balance: ${:.2}", self.position, price, self.position, self.balance);
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

    pub fn can_buy(&self, price: f64) -> bool {
        self.position == 0.0 && self.balance >= price * 0.001 // minimum 0.001 BTC
    }

    pub fn can_sell(&self) -> bool {
        self.position > 0.0
    }

    pub async fn execute_buy(&mut self, price: f64) -> Result<()> {
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
            info!("Trade P&L: ${:.2} ({:.2}%)", pnl, (pnl / (btc_to_sell * entry)) * 100.0);
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