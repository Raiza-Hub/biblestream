//! `AssemblyAI` Universal-3.5 Pro streaming STT provider.
//!
//! Connects to `AssemblyAI`'s v3 streaming WebSocket endpoint and emits the same
//! [`TranscriptEvent`] interface as the Deepgram and Whisper providers, so the
//! downstream detection pipeline is unchanged.
//!
//! Unlike [`crate::deepgram::DeepgramClient`], this provider is WebSocket-only:
//! there is no REST fallback. `AssemblyAI`'s batch (async) API is a submit-and-poll
//! flow rather than a simple buffer-and-POST window, so a hybrid mode is out of
//! scope for now.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::Receiver;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;
use url::Url;

use crate::error::SttError;
use crate::keyterms::bible_keyterms;
use crate::provider::SttProvider;
use crate::types::{SttConfig, TranscriptEvent, Word};

const MAX_RECONNECT_ATTEMPTS: u32 = 5;
const RECONNECT_DELAY: Duration = Duration::from_secs(1);
/// Batch up to 250ms of audio before sending (at 16kHz, that is 4000 samples).
const BATCH_SAMPLES: usize = 4000;
/// `AssemblyAI` requires audio chunks of at least 50ms (800 samples at 16kHz).
/// Shorter sub-chunks are held in the buffer until they reach this size.
const MIN_CHUNK_SAMPLES: usize = 800;
/// `AssemblyAI` ignores individual keyterms longer than 50 characters.
const MAX_KEYTERM_LEN: usize = 50;
/// `AssemblyAI` accepts at most 100 keyterms per session.
const MAX_KEYTERMS: usize = 100;

pub struct AssemblyAiClient {
    config: SttConfig,
    cancelled: Arc<AtomicBool>,
}

impl std::fmt::Debug for AssemblyAiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssemblyAiClient")
            .field("model", &self.config.model)
            .finish_non_exhaustive()
    }
}

impl AssemblyAiClient {
    pub fn new(config: SttConfig) -> Self {
        Self {
            config,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Build the `AssemblyAI` WebSocket URL with connection parameters and keyterms.
    ///
    /// Formatting is automatic on Universal-3.5 Pro, so `format_turns` is not
    /// sent (it only applies to the Universal Streaming English/Multilingual
    /// models, where it also produces a second formatted final per turn).
    fn build_url(&self) -> Result<Url, SttError> {
        let mut url = Url::parse("wss://streaming.assemblyai.com/v3/ws")
            .map_err(|e| SttError::ConnectionFailed(e.to_string()))?;

        {
            let mut q = url.query_pairs_mut();
            q.append_pair("speech_model", &self.config.model);
            q.append_pair("encoding", &self.config.encoding);
            q.append_pair("sample_rate", &self.config.sample_rate.to_string());
            q.append_pair("include_partial_turns", "true");

            if let Some(ref lang) = self.config.language {
                let steering = format!("[\"{lang}\"]");
                q.append_pair("language_codes", &steering);
            }

            // Keyterm boosting: core terms first, then bible_keyterms(), deduped,
            // capped at 100, dropping any term longer than 50 characters.
            let core_terms = vec![
                "Jesus".to_string(),
                "Christ".to_string(),
                "God".to_string(),
                "Lord".to_string(),
                "Holy Spirit".to_string(),
            ];
            let bible_terms = bible_keyterms();

            let mut seen = std::collections::HashSet::new();
            let mut all_keyterms: Vec<String> = Vec::new();
            for term in core_terms.into_iter().chain(bible_terms) {
                if term.chars().count() > MAX_KEYTERM_LEN {
                    continue;
                }
                if seen.insert(term.clone()) {
                    all_keyterms.push(term);
                }
                if all_keyterms.len() >= MAX_KEYTERMS {
                    break;
                }
            }

            if !all_keyterms.is_empty() {
                let keyterms_json = serde_json::to_string(&all_keyterms)
                    .map_err(|e| SttError::ConnectionFailed(e.to_string()))?;
                q.append_pair("keyterms_prompt", &keyterms_json);
                log::info!(
                    "AssemblyAI keyterm boosting: {} keyterms added",
                    all_keyterms.len()
                );
            }
        }

        log::info!("AssemblyAI WebSocket URL: {}", url.as_str());
        Ok(url)
    }

    /// Connect to `AssemblyAI` and stream audio from `audio_rx`, emitting events to `event_tx`.
    pub async fn connect(
        &self,
        audio_rx: Receiver<Vec<i16>>,
        event_tx: mpsc::Sender<TranscriptEvent>,
    ) -> Result<(), SttError> {
        if self.config.api_key.is_empty() {
            return Err(SttError::ApiKeyMissing);
        }

        let cancelled = self.cancelled.clone();
        let mut attempts: u32 = 0;

        loop {
            if cancelled.load(Ordering::SeqCst) {
                log::info!("AssemblyAiClient: cancelled, stopping connection loop");
                break;
            }

            match self
                .try_connect(audio_rx.clone(), event_tx.clone(), cancelled.clone())
                .await
            {
                Ok(()) => {
                    log::info!("AssemblyAiClient: connection closed normally");
                    break;
                }
                Err(e) => {
                    attempts += 1;
                    log::warn!(
                        "AssemblyAiClient: connection error (attempt {attempts}/{MAX_RECONNECT_ATTEMPTS}): {e}",
                    );

                    let _ = event_tx.send(TranscriptEvent::Disconnected).await;

                    if attempts >= MAX_RECONNECT_ATTEMPTS {
                        log::error!("AssemblyAiClient: max reconnection attempts reached");
                        let _ = event_tx
                            .send(TranscriptEvent::Error(format!(
                                "Max reconnection attempts reached: {e}"
                            )))
                            .await;
                        return Err(e);
                    }

                    tokio::time::sleep(RECONNECT_DELAY).await;
                }
            }
        }

        Ok(())
    }

    /// Attempt a single WebSocket connection and run send/receive loops.
    #[allow(clippy::too_many_lines)]
    async fn try_connect(
        &self,
        audio_rx: Receiver<Vec<i16>>,
        event_tx: mpsc::Sender<TranscriptEvent>,
        cancelled: Arc<AtomicBool>,
    ) -> Result<(), SttError> {
        let url = self.build_url()?;

        let mut request = url
            .as_str()
            .into_client_request()
            .map_err(|e| SttError::ConnectionFailed(e.to_string()))?;

        // AssemblyAI expects the raw API key with no `Bearer`/`Token` prefix.
        request.headers_mut().insert(
            "Authorization",
            HeaderValue::from_str(&self.config.api_key)
                .map_err(|e| SttError::ConnectionFailed(e.to_string()))?,
        );

        let (ws_stream, _response) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| SttError::ConnectionFailed(e.to_string()))?;

        log::info!("AssemblyAiClient: connected to AssemblyAI");
        let _ = event_tx.send(TranscriptEvent::Connected).await;

        let (mut write, mut read) = ws_stream.split();

        let send_cancelled = cancelled.clone();
        let recv_cancelled = cancelled.clone();

        // Track unexpected disconnects so try_connect returns Err and triggers reconnection.
        let send_error_flag = Arc::new(AtomicBool::new(false));
        let recv_error_flag = Arc::new(AtomicBool::new(false));
        let send_err = send_error_flag.clone();
        let recv_err = recv_error_flag.clone();

        // Bridge channel: blocking audio reader → async WebSocket writer
        #[allow(clippy::items_after_statements)]
        enum WsCommand {
            Audio(Vec<u8>),
            Close,
        }
        let (ws_tx, mut ws_rx) = tokio::sync::mpsc::channel::<WsCommand>(64);

        // Part 1: Blocking thread reads audio from crossbeam channel.
        // AssemblyAI has no client KeepAlive message; the continuous mic stream
        // (including silence) keeps the session alive.
        let audio_reader = {
            let ws_tx = ws_tx.clone();
            let cancelled = send_cancelled.clone();
            tokio::task::spawn_blocking(move || {
                let mut batch_buf: Vec<u8> = Vec::with_capacity(BATCH_SAMPLES * 2);
                let batch_byte_threshold = BATCH_SAMPLES * 2;
                // Chunks must be at least 50ms (800 samples → 1600 bytes).
                let min_chunk_bytes = MIN_CHUNK_SAMPLES * 2;

                loop {
                    if cancelled.load(Ordering::SeqCst) {
                        let _ = ws_tx.blocking_send(WsCommand::Close);
                        break;
                    }

                    match audio_rx.recv_timeout(Duration::from_millis(50)) {
                        Ok(samples) => {
                            for sample in &samples {
                                batch_buf.extend_from_slice(&sample.to_le_bytes());
                            }
                            if batch_buf.len() >= batch_byte_threshold {
                                let data = std::mem::take(&mut batch_buf);
                                if ws_tx.blocking_send(WsCommand::Audio(data)).is_err() {
                                    break;
                                }
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            // Only flush if the buffer already meets the 50ms
                            // minimum; otherwise hold it until more audio arrives.
                            if batch_buf.len() >= min_chunk_bytes {
                                let data = std::mem::take(&mut batch_buf);
                                if ws_tx.blocking_send(WsCommand::Audio(data)).is_err() {
                                    break;
                                }
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                            // Audio source closed — flush whatever remains and close.
                            if !batch_buf.is_empty() {
                                let data = std::mem::take(&mut batch_buf);
                                let _ = ws_tx.blocking_send(WsCommand::Audio(data));
                            }
                            let _ = ws_tx.blocking_send(WsCommand::Close);
                            break;
                        }
                    }
                }
            })
        };

        // Part 2: Async task writes to WebSocket (non-blocking, doesn't starve tokio)
        let ws_writer = tokio::spawn(async move {
            while let Some(cmd) = ws_rx.recv().await {
                match cmd {
                    WsCommand::Audio(data) => {
                        if let Err(e) = write.send(Message::Binary(data.into())).await {
                            log::error!("AssemblyAiClient ws_writer: send error: {e}");
                            send_err.store(true, Ordering::SeqCst);
                            break;
                        }
                    }
                    WsCommand::Close => {
                        let close_msg = serde_json::json!({"type": "Terminate"}).to_string();
                        let _ = write.send(Message::Text(close_msg.into())).await;
                        let _ = write.close().await;
                        break;
                    }
                }
            }
        });

        // Receiver task: reads text frames and parses AssemblyAI JSON.
        let receiver = tokio::spawn(async move {
            while let Some(msg_result) = read.next().await {
                if recv_cancelled.load(Ordering::SeqCst) {
                    break;
                }

                match msg_result {
                    Ok(Message::Text(text)) => {
                        if let Err(e) = parse_and_send(&text, &event_tx).await {
                            log::warn!("AssemblyAiClient receiver: parse error: {e}");
                        }
                    }
                    Ok(Message::Close(_)) => {
                        log::info!("AssemblyAiClient receiver: server closed connection");
                        break;
                    }
                    Ok(_) => {
                        // Ignore binary/ping/pong frames
                    }
                    Err(e) => {
                        log::error!("AssemblyAiClient receiver: WebSocket error: {e}");
                        recv_err.store(true, Ordering::SeqCst);
                        let _ = event_tx
                            .send(TranscriptEvent::Error(format!("WebSocket error: {e}")))
                            .await;
                        break;
                    }
                }
            }
        });

        // Wait for all tasks
        let _ = tokio::join!(audio_reader, ws_writer, receiver);

        // If either side had an unexpected error, return Err so the connection loop retries.
        if send_error_flag.load(Ordering::SeqCst) || recv_error_flag.load(Ordering::SeqCst) {
            return Err(SttError::ConnectionFailed(
                "Connection lost unexpectedly".into(),
            ));
        }

        Ok(())
    }

    /// Cancel the current connection and signal shutdown.
    pub fn stop(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

/// Parse an `AssemblyAI` JSON message and send the corresponding `TranscriptEvent`.
#[expect(clippy::cast_precision_loss, reason = "word counts are small")]
async fn parse_and_send(
    text: &str,
    event_tx: &mpsc::Sender<TranscriptEvent>,
) -> Result<(), SttError> {
    let json: serde_json::Value =
        serde_json::from_str(text).map_err(|e| SttError::ParseError(e.to_string()))?;

    let msg_type = json
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    match msg_type {
        "Begin" => {
            // Session accepted; the provider already emits Connected on connect,
            // but forward it here too in case of a later Begin.
            let _ = event_tx.send(TranscriptEvent::Connected).await;
            return Ok(());
        }
        "SpeechStarted" => {
            let _ = event_tx.send(TranscriptEvent::SpeechStarted).await;
            return Ok(());
        }
        "Termination" => {
            let _ = event_tx.send(TranscriptEvent::Disconnected).await;
            return Ok(());
        }
        "Error" => {
            let error = json
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            let _ = event_tx
                .send(TranscriptEvent::Error(error.to_string()))
                .await;
            return Ok(());
        }
        "Turn" => { /* continue parsing below */ }
        _ => {
            // Heartbeat, metadata, etc. — ignore silently.
            return Ok(());
        }
    }

    let end_of_turn = json
        .get("end_of_turn")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    let transcript = json
        .get("transcript")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();

    let words = json
        .get("words")
        .and_then(|w| w.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|w| {
                    // AssemblyAI reports word timestamps in milliseconds.
                    Some(Word {
                        text: w.get("text")?.as_str()?.to_string(),
                        start: w.get("start")?.as_f64()? / 1000.0,
                        end: w.get("end")?.as_f64()? / 1000.0,
                        confidence: w.get("confidence")?.as_f64()?,
                        punctuated_word: None,
                    })
                })
                .collect::<Vec<Word>>()
        })
        .unwrap_or_default();

    let event = if end_of_turn {
        // No single transcript-level confidence is provided; average word confidences.
        let confidence = if words.is_empty() {
            0.0
        } else {
            words.iter().map(|w| w.confidence).sum::<f64>() / words.len() as f64
        };
        TranscriptEvent::Final {
            transcript,
            words,
            confidence,
            speech_final: true,
        }
    } else {
        TranscriptEvent::Partial { transcript, words }
    };

    event_tx
        .send(event)
        .await
        .map_err(|e| SttError::SendError(e.to_string()))?;

    Ok(())
}

// ── SttProvider implementation ───────────────────────────────────────────────

#[async_trait::async_trait]
impl SttProvider for AssemblyAiClient {
    async fn start(
        &self,
        audio_rx: Receiver<Vec<i16>>,
        event_tx: mpsc::Sender<TranscriptEvent>,
    ) -> Result<(), SttError> {
        // WebSocket-only: no REST fallback for AssemblyAI.
        self.connect(audio_rx, event_tx).await
    }

    fn stop(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    fn name(&self) -> &'static str {
        "assemblyai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SttConfig {
        SttConfig {
            api_key: "test-key".to_string(),
            model: "universal-3-5-pro".to_string(),
            sample_rate: 16_000,
            encoding: "pcm_s16le".to_string(),
            language: None,
        }
    }

    #[test]
    fn build_url_includes_required_params() {
        let client = AssemblyAiClient::new(config());
        let url = client.build_url().expect("url builds");
        let query = url.query().unwrap_or_default();

        assert!(url.as_str().starts_with("wss://streaming.assemblyai.com/v3/ws"));
        assert!(query.contains("speech_model=universal-3-5-pro"));
        assert!(query.contains("encoding=pcm_s16le"));
        assert!(query.contains("sample_rate=16000"));
        assert!(query.contains("include_partial_turns=true"));
        assert!(query.contains("keyterms_prompt="));
        // Universal-3.5 Pro formats automatically; format_turns must not be sent.
        assert!(!query.contains("format_turns"));
        // The API key is never placed in the URL.
        assert!(!query.contains("test-key"));
    }

    #[test]
    fn keyterms_are_capped_and_short() {
        let client = AssemblyAiClient::new(config());
        let url = client.build_url().expect("url builds");
        let keyterms = url
            .query_pairs()
            .find(|(k, _)| k == "keyterms_prompt")
            .map(|(_, v)| v.into_owned())
            .expect("keyterms present");

        let parsed: Vec<String> = serde_json::from_str(&keyterms).expect("valid json array");
        assert_ne!(parsed.len(), 0, "expected at least one keyterm");
        assert!(parsed.len() <= MAX_KEYTERMS);
        assert!(parsed.iter().all(|t| t.chars().count() <= MAX_KEYTERM_LEN));
    }

    #[tokio::test]
    async fn parses_final_turn_with_millisecond_word_timestamps() {
        let (tx, mut rx) = mpsc::channel::<TranscriptEvent>(4);
        let msg = r#"{
            "type": "Turn",
            "turn_order": 0,
            "turn_is_formatted": true,
            "end_of_turn": true,
            "transcript": "My name is Sonny.",
            "end_of_turn_confidence": 0.9,
            "words": [
                {"start": 1000, "end": 1500, "text": "My", "confidence": 0.8, "word_is_final": true},
                {"start": 1500, "end": 2000, "text": "name", "confidence": 1.0, "word_is_final": true}
            ]
        }"#;

        parse_and_send(msg, &tx).await.expect("parses");

        match rx.recv().await {
            Some(TranscriptEvent::Final {
                transcript,
                words,
                confidence,
                speech_final,
            }) => {
                assert_eq!(transcript, "My name is Sonny.");
                assert_eq!(words.len(), 2);
                // Milliseconds are converted to seconds.
                assert!((words[0].start - 1.0).abs() < f64::EPSILON);
                assert!((words[1].end - 2.0).abs() < f64::EPSILON);
                assert!((confidence - 0.9).abs() < 1e-9);
                assert!(speech_final);
            }
            other => panic!("expected Final, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn parses_partial_turn() {
        let (tx, mut rx) = mpsc::channel::<TranscriptEvent>(4);
        let msg = r#"{
            "type": "Turn",
            "turn_order": 0,
            "turn_is_formatted": false,
            "end_of_turn": false,
            "transcript": "modern medicine is",
            "words": []
        }"#;

        parse_and_send(msg, &tx).await.expect("parses");

        match rx.recv().await {
            Some(TranscriptEvent::Partial { transcript, .. }) => {
                assert_eq!(transcript, "modern medicine is");
            }
            other => panic!("expected Partial, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_speech_started_and_termination() {
        let (tx, mut rx) = mpsc::channel::<TranscriptEvent>(4);

        parse_and_send(r#"{"type":"SpeechStarted","timestamp":0}"#, &tx)
            .await
            .expect("parses");
        assert!(matches!(rx.recv().await, Some(TranscriptEvent::SpeechStarted)));

        parse_and_send(r#"{"type":"Termination","audio_duration_seconds":1}"#, &tx)
            .await
            .expect("parses");
        assert!(matches!(
            rx.recv().await,
            Some(TranscriptEvent::Disconnected)
        ));
    }
}
