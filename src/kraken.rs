use anyhow::Result;
use futures::stream::{SelectAll, StreamExt};
use futures::SinkExt;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

#[derive(Debug, Clone)]
pub struct KrakenTrade {
    pub price: f64,
    pub volume: f64,
    pub timestamp: u64,
    pub pair: String,
}

#[derive(Serialize)]
struct KrakenSubscribeMessage {
    event: String,
    pair: Vec<String>,
    subscription: KrakenSubscription,
}

#[derive(Serialize)]
struct KrakenSubscription {
    name: String,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct KrakenEventMessage {
    event: String,
    status: Option<String>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct KrakenTradeMessage {
    #[serde(rename = "0")] // First element is the channel ID
    channel_id: u32,
    #[serde(rename = "1")] // Second element is the trade data array
    trades: Vec<Vec<serde_json::Value>>,
    #[serde(rename = "2")] // Third element is the channel name
    channel: String,
    #[serde(rename = "3")] // Fourth element is the pair
    pair: String,
}

pub struct KrakenStream {
    streams: SelectAll<Pin<Box<dyn Stream<Item = Result<KrakenTrade>> + Send>>>,
}

impl KrakenStream {
    pub async fn connect(pairs: Vec<String>) -> Result<Self> {
        let mut streams = SelectAll::new();

        for pair in pairs {
            let stream = Self::connect_pair(&pair).await?;
            streams.push(stream);
        }

        Ok(KrakenStream { streams })
    }

    async fn connect_pair(
        pair: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<KrakenTrade>> + Send>>> {
        let pair = pair.to_string(); // Clone the pair string
        let url = "wss://ws.kraken.com";

        let (ws_stream, _) = connect_async(url).await?;
        let (write, read) = ws_stream.split();

        // Wrap the write half in Arc<Mutex<>> for shared access
        let write = Arc::new(Mutex::new(write));

        // Send subscription message
        let subscribe_msg = KrakenSubscribeMessage {
            event: "subscribe".to_string(),
            pair: vec![pair.to_string()],
            subscription: KrakenSubscription {
                name: "trade".to_string(),
            },
        };

        let msg = serde_json::to_string(&subscribe_msg)?;
        write.lock().await.send(Message::Text(msg)).await?;

        // Create a stream that processes messages
        let stream = read.filter_map(move |message| {
            let write = Arc::clone(&write);
            let pair = pair.clone(); // Clone for the closure
            async move {
                match message {
                    Ok(Message::Text(text)) => Self::process_message(&text, &pair).await,
                    Ok(Message::Ping(payload)) => {
                        // Respond to ping
                        let _ = write.lock().await.send(Message::Pong(payload)).await;
                        None
                    }
                    _ => None,
                }
            }
        });

        Ok(Box::pin(stream))
    }

    async fn process_message(text: &str, pair: &str) -> Option<Result<KrakenTrade>> {
        // Try to parse as trade message
        if let Ok(trade_msg) = serde_json::from_str::<KrakenTradeMessage>(text) {
            if let Some(trade_data) = trade_msg.trades.first() {
                if trade_data.len() >= 6 {
                    // Kraken trade format: [price, volume, time, side, order_type, misc]
                    if let (Some(price_str), Some(volume_str), Some(time_str)) = (
                        trade_data[0].as_str(),
                        trade_data[1].as_str(),
                        trade_data[2].as_f64(),
                    ) {
                        if let (Ok(price), Ok(volume)) =
                            (price_str.parse::<f64>(), volume_str.parse::<f64>())
                        {
                            return Some(Ok(KrakenTrade {
                                price,
                                volume,
                                timestamp: (time_str * 1000.0) as u64, // Convert to milliseconds
                                pair: pair.to_string(),
                            }));
                        }
                    }
                }
            }
        }

        // Try to parse as event message (subscription confirmation)
        if serde_json::from_str::<KrakenEventMessage>(text).is_ok() {
            // Subscription event, ignore
            return None;
        }

        None
    }

    pub async fn next(&mut self) -> Option<Result<KrakenTrade>> {
        self.streams.next().await
    }
}
