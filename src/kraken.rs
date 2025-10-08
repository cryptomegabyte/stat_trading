use anyhow::Result;
use async_stream::stream;
use futures::stream::{SelectAll, StreamExt};
use futures::SinkExt;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout};
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

#[derive(Debug)]
struct CircuitBreaker {
    failures: u32,
    last_failure: std::time::Instant,
    state: CircuitState,
}

#[derive(Debug, Clone, Copy)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    fn new() -> Self {
        Self {
            failures: 0,
            last_failure: std::time::Instant::now(),
            state: CircuitState::Closed,
        }
    }

    fn record_success(&mut self) {
        self.failures = 0;
        self.state = CircuitState::Closed;
    }

    fn record_failure(&mut self) {
        self.failures += 1;
        self.last_failure = std::time::Instant::now();

        if self.failures >= 5 {
            self.state = CircuitState::Open;
        }
    }

    fn should_attempt(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout has passed (30 seconds)
                if self.last_failure.elapsed() > Duration::from_secs(30) {
                    self.state = CircuitState::HalfOpen;
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct KrakenEventMessage {
    event: String,
    status: Option<String>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct KrakenTradeMessage(Vec<serde_json::Value>);

impl KrakenTradeMessage {
    fn trades(&self) -> Option<&Vec<serde_json::Value>> {
        self.0.get(1)?.as_array()
    }
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
        let pair = pair.to_string();
        let circuit_breaker = Arc::new(Mutex::new(CircuitBreaker::new()));

        let stream = stream! {
            loop {
                let should_attempt = circuit_breaker.lock().await.should_attempt();

                if !should_attempt {
                    sleep(Duration::from_secs(5)).await;
                    continue;
                }

                match Self::establish_connection(&pair, Arc::clone(&circuit_breaker)).await {
                    Ok(connection_stream) => {
                        circuit_breaker.lock().await.record_success();
                        for await item in connection_stream {
                            yield item;
                        }
                    }
                    Err(e) => {
                        circuit_breaker.lock().await.record_failure();
                        tracing::warn!("Connection failed for {}: {}. Retrying...", pair, e);
                        sleep(Duration::from_secs(5)).await;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    async fn establish_connection(
        pair: &str,
        circuit_breaker: Arc<Mutex<CircuitBreaker>>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<KrakenTrade>> + Send>>> {
        let url = "wss://ws.kraken.com";

        // Add connection timeout
        let connection_future = connect_async(url);
        let (ws_stream, _) = timeout(Duration::from_secs(10), connection_future)
            .await
            .map_err(|_| anyhow::anyhow!("Connection timeout"))??;

        let (write, read) = ws_stream.split();
        let write = Arc::new(Mutex::new(write));

        // Send subscription message with timeout
        let subscribe_msg = KrakenSubscribeMessage {
            event: "subscribe".to_string(),
            pair: vec![pair.to_string()],
            subscription: KrakenSubscription {
                name: "trade".to_string(),
            },
        };

        let msg = serde_json::to_string(&subscribe_msg)?;
        timeout(
            Duration::from_secs(5),
            write.lock().await.send(Message::Text(msg))
        ).await
        .map_err(|_| anyhow::anyhow!("Subscription timeout"))??;

        // Create a stream that processes messages with error handling
        let pair_clone = pair.to_string();
        let stream = read.filter_map(move |message| {
            let write = Arc::clone(&write);
            let pair = pair_clone.clone();
            let circuit_breaker = Arc::clone(&circuit_breaker);

            async move {
                match message {
                    Ok(Message::Text(text)) => Self::process_message(&text, &pair).await,
                    Ok(Message::Ping(payload)) => {
                        // Respond to ping with timeout
                        let pong_result = timeout(
                            Duration::from_secs(5),
                            write.lock().await.send(Message::Pong(payload))
                        ).await;

                        if pong_result.is_err() {
                            circuit_breaker.lock().await.record_failure();
                        }
                        None
                    }
                    Ok(Message::Close(_)) => {
                        circuit_breaker.lock().await.record_failure();
                        None
                    }
                    Err(e) => {
                        circuit_breaker.lock().await.record_failure();
                        Some(Err(anyhow::anyhow!("WebSocket error: {}", e)))
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
            if let Some(trade_data_array) = trade_msg.trades() {
                for trade_data in trade_data_array {
                    if let Some(trade_array) = trade_data.as_array() {
                        if trade_array.len() >= 6 {
                            // Kraken trade format: [price, volume, time, side, order_type, misc]
                            if let (Some(price_str), Some(volume_str), Some(time_str)) = (
                                trade_array[0].as_str(),
                                trade_array[1].as_str(),
                                trade_array[2].as_str(), // Time is a string, not f64
                            ) {
                                if let (Ok(price), Ok(volume)) =
                                    (price_str.parse::<f64>(), volume_str.parse::<f64>())
                                {
                                    if let Ok(time_val) = time_str.parse::<f64>() {
                                        return Some(Ok(KrakenTrade {
                                            price,
                                            volume,
                                            timestamp: (time_val * 1000.0) as u64, // Convert to milliseconds
                                            pair: pair.to_string(),
                                        }));
                                    }
                                }
                            }
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
