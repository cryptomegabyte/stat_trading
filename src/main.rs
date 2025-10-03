mod types;
mod ml;
mod trading;

use anyhow::Result;
use barter_data::exchange::binance::spot::BinanceSpot;
use barter_data::streams::Streams;
use barter_data::subscription::trade::PublicTrades;
use barter_instrument::instrument::market_data::kind::MarketDataInstrumentKind;
use futures::StreamExt;
use std::env;
use tracing::{info, warn, Level};
use trading::{Backtester, LiveTrader};




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
        if let barter_data::streams::reconnect::Event::Item(result) = event {
            match result {
                Ok(trade) => {
                    let price = trade.kind.price;
                    let _volume = trade.kind.amount;

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
    
    for _i in 0..1000 {
        // Random walk with small steps
        let change = (rand::random::<f64>() - 0.5) * 100.0; // +/- 50
        price += change;
        price = price.clamp(40000.0, 60000.0); // bound it
        
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