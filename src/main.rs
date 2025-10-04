pub mod types;
pub mod ml;
pub mod trading;

use anyhow::Result;
use barter_data::exchange::binance::spot::BinanceSpot;
use barter_data::streams::Streams;
use barter_data::subscription::trade::PublicTrades;
use barter_instrument::instrument::market_data::kind::MarketDataInstrumentKind;
use futures::StreamExt;
use std::env;
use tracing::{info, warn, Level};
use trading::{Backtester, LiveTrader};
use types::TradingPair;




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
    info!("🚀 Starting MULTI-PAIR LIVE TRADING BOT");
    info!("⚠️  WARNING: This will execute real trades!");
    info!("💰 Starting balance: $2000 total (shared between all pairs)");
    info!("🎯 Risk management: 1.5% stop-loss, 4% take-profit, 10% position sizing");
    info!("📊 Trading pairs: XRP, BNB");

    // Check for API keys (optional for now - will simulate)
    let api_key = env::var("BINANCE_API_KEY").ok();
    let secret_key = env::var("BINANCE_SECRET_KEY").ok();

    if api_key.is_none() || secret_key.is_none() {
        info!("⚠️  No API keys found - running in SIMULATION mode");
        info!("💡 Set BINANCE_API_KEY and BINANCE_SECRET_KEY for real trading");
    } else {
        info!("🔑 API keys found - ready for LIVE trading");
    }

    // Initialize market streams for top performing pairs
    let subscriptions: Vec<_> = vec![
        (BinanceSpot::default(), "xrp", "usdt", MarketDataInstrumentKind::Spot, PublicTrades),
        (BinanceSpot::default(), "bnb", "usdt", MarketDataInstrumentKind::Spot, PublicTrades),
    ];

    let mut streams = Streams::<PublicTrades>::builder()
        .subscribe(subscriptions)
        .init()
        .await?;

    // Initialize live trader
    let mut trader = LiveTrader::new();

    // Select the stream for Binance
    let mut binance_stream = streams
        .select(barter_instrument::exchange::ExchangeId::BinanceSpot)
        .unwrap();

    info!("📡 Connected to Binance streams for XRP, BNB");
    info!("🤖 Bot is now monitoring markets and generating signals...");

    // Status update counter
    let mut status_counter = 0;
    let mut trade_count = 0;

    // Main trading loop
    while let Some(event) = binance_stream.next().await {
        if let barter_data::streams::reconnect::Event::Item(result) = event {
            match result {
                Ok(trade) => {
                    let price = trade.kind.price;
                    let volume = trade.kind.amount;

                    // Identify which pair this trade is for
                    let pair = match trade.instrument.base.to_string().as_str() {
                        "xrp" => TradingPair::XRP,
                        "bnb" => TradingPair::BNB,
                        _ => continue, // Skip unknown pairs
                    };

                    // Process price update and check for trading signals
                    trader.process_price_update(&pair, price, volume).await?;

                    trade_count += 1;
                    status_counter += 1;

                    // Print status every 500 trades (less frequent with multiple pairs)
                    if status_counter >= 500 {
                        trader.print_status();
                        status_counter = 0;
                    }

                    // Safety check - stop after 5000 trades for demo (5x more with 5 pairs)
                    if trade_count >= 5000 {
                        info!("🛑 Demo limit reached (5000 trades) - stopping bot");
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

    // Note: In multi-pair trading, positions are managed per pair
    // No single position to close at the end

    Ok(())
}

async fn run_backtest() -> Result<()> {
    info!("🚀 Starting MULTI-PAIR BACKTEST");
    info!("📊 Testing pairs: XRP, BNB");
    info!("📅 1 year of hourly data from Kraken");

    // Initialize backtester
    let mut backtester = Backtester::new();

    // Fetch data for each pair
    let pairs = backtester.config.pairs.clone(); // Clone to avoid borrowing issues
    for pair in &pairs {
        info!("📡 Fetching {} data from Kraken...", pair.symbol().to_uppercase());

        let pair_data = fetch_pair_data(pair).await?;
        info!("✅ Fetched {} data points for {}", pair_data.len(), pair.symbol().to_uppercase());

        // Process trades for this pair
        for (price, volume) in &pair_data {
            backtester.process_trade(pair, *price, *volume);
        }

        // Close any open position for this pair at market close
        if let Some(trader) = backtester.traders.get_mut(pair) {
            if trader.position > 0.0 {
                let last_price = pair_data.last().unwrap().0;
                trader.sell(last_price);
                info!("💰 Closed {} position at end of backtest at ${}", pair.symbol().to_uppercase(), last_price);
            }
        }
    }

    // Print results
    backtester.print_results();

    Ok(())
}

async fn fetch_pair_data(pair: &TradingPair) -> Result<Vec<(f64, f64)>> {
    let client = reqwest::Client::new();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let now_seconds = now / 1000;
    let mut last_time = now_seconds - 365 * 24 * 60 * 60; // 1 year ago
    let mut all_ohlc = Vec::new();

    loop {
        let url = format!("https://api.kraken.com/0/public/OHLC?pair={}&interval=60&since={}", pair.kraken_pair(), last_time);
        let response = client.get(&url).send().await?;
        let json: serde_json::Value = response.json().await?;

        if let Some(result) = json["result"].as_object() {
            if let Some(pair_data) = result.get(pair.kraken_symbol()) {
                if let Some(ohlc_array) = pair_data.as_array() {
                    for ohlc in ohlc_array {
                        if let Some(arr) = ohlc.as_array() {
                            all_ohlc.push(arr.clone());
                        }
                    }
                    // Get last time for next request
                    if let Some(last) = ohlc_array.last() {
                        if let Some(time_str) = last.as_array().and_then(|a| a.get(0)).and_then(|v| v.as_u64()) {
                            last_time = time_str;
                        }
                    }
                }
            }
        }

        // Kraken returns up to 720 points per request, stop if less than that
        if all_ohlc.len() >= 720 || json["result"][pair.kraken_symbol()].as_array().map_or(true, |a| a.len() < 720) {
            break;
        }
    }

    let mut trades = Vec::new();
    for ohlc in &all_ohlc {
        let close_price: f64 = ohlc[4].as_str().unwrap_or(&ohlc[4].to_string()).parse().unwrap();
        let volume: f64 = ohlc[6].as_str().unwrap_or(&ohlc[6].to_string()).parse().unwrap();
        trades.push((close_price, volume));
    }

    Ok(trades)
}