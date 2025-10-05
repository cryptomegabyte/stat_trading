pub mod kraken;
pub mod ml;
pub mod trading;
pub mod types;

use anyhow::Result;
use kraken::KrakenStream;
use std::env;
use tracing::{info, warn, Level};
use trading::{Backtester, LiveTrader};
use types::TradingPair;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    // Check for backtest mode and model type
    let args: Vec<String> = env::args().collect();
    let is_backtest = args.contains(&"--backtest".to_string());
    let is_paper_trading = args.contains(&"--paper-trading".to_string());

    // Default to gas_vg model (best performing from backtests)
    let model_type = if let Some(pos) = args.iter().position(|arg| arg == "--model") {
        args.get(pos + 1).unwrap_or(&"gas_vg".to_string()).clone()
    } else {
        "gas_vg".to_string()
    };

    if is_backtest {
        info!("Starting backtest mode with model: {}", model_type);
        run_backtest(model_type).await?;
    } else if is_paper_trading {
        info!(
            "Starting PAPER TRADING mode with model: {} (NO REAL MONEY)",
            model_type
        );
        run_paper_trading(model_type).await?;
    } else {
        info!("Starting live trading bot demo with Kraken and ML prediction");
        run_live().await?;
    }

    Ok(())
}

async fn run_paper_trading(model_type: String) -> Result<()> {
    info!("📈 PAPER TRADING MODE - SIMULATED TRADES ONLY");
    info!("💰 Starting balance: $2000 total (shared between all pairs)");
    info!("🎯 Risk management: 1.5% stop-loss, 4% take-profit, dynamic position sizing");
    info!("🤖 Using model: {}", model_type);
    info!("⚠️  NO REAL MONEY - This is for testing and learning only");

    // Initialize Kraken WebSocket streams for top performing pairs
    let pairs = vec!["XBT/USD".to_string(), "XRP/USD".to_string()];
    let mut kraken_stream = KrakenStream::connect(pairs).await?;

    // Initialize paper trader with specified model
    let mut trader = LiveTrader::new_with_model(&model_type);

    info!("📡 Connected to Kraken streams for XBT, XRP (PAPER TRADING)");
    info!("🤖 Bot is now monitoring markets and generating signals...");

    // Status update counter
    let mut status_counter = 0;
    let mut trade_count = 0;

    // Main trading loop
    while let Some(result) = kraken_stream.next().await {
        match result {
            Ok(trade) => {
                let price = trade.price;
                let volume = trade.volume;

                // Identify which pair this trade is for
                let pair = match trade.pair.as_str() {
                    "XBT/USD" => TradingPair::BTC,
                    "XRP/USD" => TradingPair::XRP,
                    _ => continue, // Skip unknown pairs
                };

                // Process price update and check for trading signals
                trader.process_price_update(&pair, price, volume).await?;

                trade_count += 1;
                status_counter += 1;

                // Print status every 1000 trades (more frequent for paper trading)
                if status_counter >= 1000 {
                    trader.print_paper_status();
                    status_counter = 0;
                }

                // Safety check - stop after 10000 trades for demo (10x more for paper trading)
                if trade_count >= 10000 {
                    info!("🛑 Paper trading demo limit reached (10000 trades) - stopping bot");
                    break;
                }
            }
            Err(e) => {
                warn!("❌ Error receiving trade: {:?}", e);
                // Continue on errors
            }
        }
    }

    // Final status
    info!("🏁 Paper trading session ended");
    trader.print_paper_status();

    Ok(())
}

async fn run_live() -> Result<()> {
    info!("🚀 Starting MULTI-PAIR LIVE TRADING BOT");
    info!("⚠️  WARNING: This will execute real trades!");
    info!("💰 Starting balance: $2000 total (shared between all pairs)");
    info!("🎯 Risk management: 1.5% stop-loss, 4% take-profit, 10% position sizing");
    info!("Trading pairs: XRP, BNB");

    // Check for API keys (optional for now - will simulate)
    let api_key = env::var("KRAKEN_API_KEY").ok();
    let secret_key = env::var("KRAKEN_SECRET_KEY").ok();

    if api_key.is_none() || secret_key.is_none() {
        info!("⚠️  No API keys found - running in SIMULATION mode");
        info!("💡 Set KRAKEN_API_KEY and KRAKEN_SECRET_KEY for real trading");
    } else {
        info!("🔑 API keys found - ready for LIVE trading");
    }

    // Initialize Kraken WebSocket streams for top performing pairs
    let pairs = vec!["XBT/USD".to_string(), "XRP/USD".to_string()];
    let mut kraken_stream = KrakenStream::connect(pairs).await?;

    // Initialize live trader
    let mut trader = LiveTrader::new();

    info!("📡 Connected to Kraken streams for BTC, XRP");
    info!("🤖 Bot is now monitoring markets and generating signals...");

    // Status update counter
    let mut status_counter = 0;
    let mut trade_count = 0;

    // Main trading loop
    while let Some(result) = kraken_stream.next().await {
        match result {
            Ok(trade) => {
                let price = trade.price;
                let volume = trade.volume;

                // Identify which pair this trade is for
                let pair = match trade.pair.as_str() {
                    "XBT/USD" => TradingPair::BTC,
                    "XRP/USD" => TradingPair::XRP,
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

    // Final status
    info!("🏁 Live trading session ended");
    trader.print_status();

    // Note: In multi-pair trading, positions are managed per pair
    // No single position to close at the end

    Ok(())
}

async fn run_backtest(model_type: String) -> Result<()> {
    info!("🚀 Starting MULTI-PAIR BACKTEST with model: {}", model_type);
    info!("📊 Testing multiple crypto pairs: BTC, ETH, XRP, SOL, BNB, ADA, DOT, LINK, LTC, AVAX");
    info!("📅 1 year of hourly data from Kraken");

    // Initialize backtester with specified model type
    let mut backtester = Backtester::with_model_type(&model_type);

    // Fetch data for each pair
    let pairs = backtester.config.pairs.clone(); // Clone to avoid borrowing issues
    for pair in &pairs {
        info!(
            "📡 Fetching {} data from Kraken...",
            pair.symbol().to_uppercase()
        );

        let pair_data = fetch_pair_data(pair).await?;
        info!(
            "✅ Fetched {} data points for {}",
            pair_data.len(),
            pair.symbol().to_uppercase()
        );

        // Process trades for this pair
        for (price, volume) in &pair_data {
            backtester.process_trade(pair, *price, *volume);
        }

        // Close any open position for this pair at market close
        if let Some(trader) = backtester.traders.get_mut(pair) {
            if trader.position > 0.0 {
                let last_price = pair_data.last().unwrap().0;
                trader.sell(last_price);
                info!(
                    "💰 Closed {} position at end of backtest at ${}",
                    pair.symbol().to_uppercase(),
                    last_price
                );
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
        let url = format!(
            "https://api.kraken.com/0/public/OHLC?pair={}&interval=60&since={}",
            pair.kraken_pair(),
            last_time
        );
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
                        if let Some(time_str) = last
                            .as_array()
                            .and_then(|a| a.first())
                            .and_then(|v| v.as_u64())
                        {
                            last_time = time_str;
                        }
                    }
                }
            }
        }

        // Kraken returns up to 720 points per request, stop if less than that
        if all_ohlc.len() >= 720
            || json["result"][pair.kraken_symbol()]
                .as_array()
                .is_none_or(|a| a.len() < 720)
        {
            break;
        }
    }

    let mut trades = Vec::new();
    for ohlc in &all_ohlc {
        let close_price: f64 = ohlc[4]
            .as_str()
            .unwrap_or(&ohlc[4].to_string())
            .parse()
            .unwrap();
        let volume: f64 = ohlc[6]
            .as_str()
            .unwrap_or(&ohlc[6].to_string())
            .parse()
            .unwrap();
        trades.push((close_price, volume));
    }

    Ok(trades)
}
