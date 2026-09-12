use base64::Engine;
use reqwest::{Client, Response, StatusCode, Url};
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Duration};
use thiserror::Error;
use tokio::sync::oneshot;

pub const DEFAULT_OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, Error)]
pub enum OllamaTransportError {
    #[error("Invalid Ollama endpoint: {0}")]
    InvalidEndpoint(String),
    #[error("Invalid Ollama model tag: {0}")]
    InvalidModel(String),
    #[error("Could not reach Ollama: {0}")]
    Request(String),
    #[error("Ollama returned {status}: {detail}")]
    Http { status: StatusCode, detail: String },
    #[error("Ollama returned an API error: {0}")]
    Api(String),
    #[error("Ollama returned malformed NDJSON: {0}")]
    MalformedStream(String),
    #[error("Ollama returned no assistant content")]
    EmptyResponse,
    #[error("Request cancelled")]
    Cancelled,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

impl OllamaChatMessage {
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            images: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaStreamMessage {
    #[serde(default)]
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaStreamChunk {
    #[serde(default)]
    pub message: Option<OllamaStreamMessage>,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaTag {
    pub name: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub modified_at: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub details: Option<OllamaModelDetails>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct OllamaModelDetails {
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub parameter_size: Option<String>,
    #[serde(default)]
    pub quantization_level: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaTagsResponse {
    #[serde(default)]
    pub models: Vec<OllamaTag>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaShowResponse {
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub details: Option<OllamaModelDetails>,
    #[serde(default)]
    pub modified_at: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OllamaClient {
    client: Client,
    base_url: String,
    bearer_token: Option<String>,
}

impl OllamaClient {
    pub fn new(base_url: &str, bearer_token: Option<&str>) -> Result<Self, OllamaTransportError> {
        let base_url = normalize_base_url(base_url)?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|error| OllamaTransportError::Request(error.to_string()))?;
        Ok(Self {
            client,
            base_url,
            bearer_token: bearer_token
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}/api/{}", self.base_url, path.trim_start_matches('/'))
    }

    fn authenticated(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.bearer_token.as_deref() {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    async fn response_detail(response: Response, bearer_token: Option<&str>) -> String {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let body = body.trim();
        if body.is_empty() {
            status.to_string()
        } else {
            sanitize_detail_with_secret(body, bearer_token)
        }
    }

    async fn check_response(&self, response: Response) -> Result<Response, OllamaTransportError> {
        if response.status().is_success() {
            Ok(response)
        } else {
            let status = response.status();
            let detail = Self::response_detail(response, self.bearer_token.as_deref()).await;
            Err(OllamaTransportError::Http { status, detail })
        }
    }

    pub async fn tags(&self) -> Result<Vec<OllamaTag>, OllamaTransportError> {
        let response = self
            .authenticated(self.client.get(self.endpoint("tags")))
            .timeout(DISCOVERY_TIMEOUT)
            .send()
            .await
            .map_err(|error| OllamaTransportError::Request(sanitize_detail(&error.to_string())))?;
        let response = self.check_response(response).await?;
        response
            .json::<OllamaTagsResponse>()
            .await
            .map(|value| value.models)
            .map_err(|error| {
                OllamaTransportError::MalformedStream(sanitize_detail(&error.to_string()))
            })
    }

    pub async fn show(&self, model: &str) -> Result<OllamaShowResponse, OllamaTransportError> {
        validate_model_tag(model)?;
        let response = self
            .authenticated(self.client.post(self.endpoint("show")))
            .timeout(DISCOVERY_TIMEOUT)
            .json(&serde_json::json!({ "model": model }))
            .send()
            .await
            .map_err(|error| OllamaTransportError::Request(sanitize_detail(&error.to_string())))?;
        let response = self.check_response(response).await?;
        response
            .json::<OllamaShowResponse>()
            .await
            .map_err(|error| {
                OllamaTransportError::MalformedStream(sanitize_detail(&error.to_string()))
            })
    }

    pub async fn complete_chat(
        &self,
        request: &OllamaChatRequest,
        cancel_rx: &mut oneshot::Receiver<()>,
    ) -> Result<String, OllamaTransportError> {
        self.complete_chat_inner(request, Some(cancel_rx)).await
    }

    pub async fn complete_chat_uncancelled(
        &self,
        request: &OllamaChatRequest,
    ) -> Result<String, OllamaTransportError> {
        self.complete_chat_inner(request, None).await
    }

    async fn complete_chat_inner(
        &self,
        request: &OllamaChatRequest,
        mut cancel_rx: Option<&mut oneshot::Receiver<()>>,
    ) -> Result<String, OllamaTransportError> {
        let mut request = request.clone();
        request.stream = false;
        let response = match cancel_rx.as_mut() {
            Some(cancel_rx) => tokio::select! {
                _ = &mut **cancel_rx => return Err(OllamaTransportError::Cancelled),
                result = self.authenticated(self.client.post(self.endpoint("chat")))
                    .json(&request)
                    .send() => result,
            },
            None => {
                self.authenticated(self.client.post(self.endpoint("chat")))
                    .json(&request)
                    .send()
                    .await
            }
        }
        .map_err(|error| OllamaTransportError::Request(sanitize_detail(&error.to_string())))?;
        let response = self.check_response(response).await?;
        let value: OllamaStreamChunk = match cancel_rx.as_mut() {
            Some(cancel_rx) => tokio::select! {
                _ = &mut **cancel_rx => return Err(OllamaTransportError::Cancelled),
                result = response.json() => result,
            },
            None => response.json().await,
        }
        .map_err(|error| {
            OllamaTransportError::MalformedStream(sanitize_detail(&error.to_string()))
        })?;
        if let Some(error) = value.error {
            return Err(OllamaTransportError::Api(sanitize_detail_with_secret(
                &error,
                self.bearer_token.as_deref(),
            )));
        }
        let content = value
            .message
            .map(|message| message.content)
            .unwrap_or_default();
        if content.trim().is_empty() {
            return Err(OllamaTransportError::EmptyResponse);
        }
        Ok(content)
    }

    pub async fn stream_chat<F>(
        &self,
        request: &OllamaChatRequest,
        cancel_rx: &mut oneshot::Receiver<()>,
        mut on_delta: F,
    ) -> Result<String, OllamaTransportError>
    where
        F: FnMut(&str),
    {
        let mut request = request.clone();
        request.stream = true;
        let response = tokio::select! {
            _ = &mut *cancel_rx => return Err(OllamaTransportError::Cancelled),
            result = self.authenticated(self.client.post(self.endpoint("chat")))
                .json(&request)
                .send() => result,
        }
        .map_err(|error| OllamaTransportError::Request(sanitize_detail(&error.to_string())))?;
        let mut response = self.check_response(response).await?;
        let mut pending = Vec::new();
        let mut content = String::new();
        let mut saw_done = false;
        loop {
            let chunk = tokio::select! {
                _ = &mut *cancel_rx => return Err(OllamaTransportError::Cancelled),
                result = response.chunk() => result,
            }
            .map_err(|error| OllamaTransportError::Request(sanitize_detail(&error.to_string())))?;
            let Some(chunk) = chunk else { break };
            pending.extend_from_slice(&chunk);
            while let Some(position) = pending.iter().position(|byte| *byte == b'\n') {
                let line = pending.drain(..=position).collect::<Vec<_>>();
                let line = &line[..line.len() - 1];
                if line.iter().all(|byte| byte.is_ascii_whitespace()) {
                    continue;
                }
                let line = String::from_utf8(line.to_vec()).map_err(|error| {
                    OllamaTransportError::MalformedStream(sanitize_detail(&error.to_string()))
                })?;
                let parsed = parse_ndjson_line(&line)?;
                if let Some(error) = parsed.error {
                    return Err(OllamaTransportError::Api(sanitize_detail_with_secret(
                        &error,
                        self.bearer_token.as_deref(),
                    )));
                }
                if let Some(message) = parsed.message {
                    if !message.content.is_empty() {
                        content.push_str(&message.content);
                        on_delta(&message.content);
                    }
                }
                saw_done |= parsed.done;
            }
        }
        if !pending.iter().all(|byte| byte.is_ascii_whitespace()) {
            let line = String::from_utf8(std::mem::take(&mut pending)).map_err(|error| {
                OllamaTransportError::MalformedStream(sanitize_detail(&error.to_string()))
            })?;
            let parsed = parse_ndjson_line(&line)?;
            if let Some(error) = parsed.error {
                return Err(OllamaTransportError::Api(sanitize_detail_with_secret(
                    &error,
                    self.bearer_token.as_deref(),
                )));
            }
            if let Some(message) = parsed.message {
                if !message.content.is_empty() {
                    content.push_str(&message.content);
                    on_delta(&message.content);
                }
            }
            saw_done |= parsed.done;
        }
        if !saw_done && content.trim().is_empty() {
            return Err(OllamaTransportError::EmptyResponse);
        }
        if content.trim().is_empty() {
            return Err(OllamaTransportError::EmptyResponse);
        }
        Ok(content)
    }
}

pub fn normalize_base_url(value: &str) -> Result<String, OllamaTransportError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(OllamaTransportError::InvalidEndpoint(
            "Enter an Ollama endpoint URL".into(),
        ));
    }
    let mut parsed = Url::parse(value)
        .map_err(|_| OllamaTransportError::InvalidEndpoint("Enter a valid endpoint URL".into()))?;
    let host = parsed.host_str().unwrap_or_default();
    let local = matches!(host, "localhost" | "127.0.0.1" | "::1");
    if !matches!(parsed.scheme(), "http" | "https")
        || (!local && parsed.scheme() != "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(OllamaTransportError::InvalidEndpoint(
            "Use a local HTTP or remote HTTPS Ollama endpoint without credentials or query parameters".into(),
        ));
    }
    let path = parsed.path().trim_end_matches('/').to_string();
    parsed.set_path(if path == "/api" { "" } else { &path });
    Ok(parsed.to_string().trim_end_matches('/').to_string())
}

pub fn validate_model_tag(value: &str) -> Result<(), OllamaTransportError> {
    let value = value.trim();
    if value.is_empty()
        || value.len() > 200
        || value
            .chars()
            .any(|character| character.is_ascii_control() || character.is_whitespace())
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "-_.:/@".contains(character))
    {
        return Err(OllamaTransportError::InvalidModel(
            "Use an existing Ollama model tag such as llama3.2:latest".into(),
        ));
    }
    Ok(())
}

pub fn encode_image(path: &Path) -> Result<String, OllamaTransportError> {
    let bytes =
        std::fs::read(path).map_err(|error| OllamaTransportError::Request(error.to_string()))?;
    if bytes.is_empty() {
        return Err(OllamaTransportError::Request(
            "Reference image is empty".into(),
        ));
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

pub fn parse_ndjson_line(line: &str) -> Result<OllamaStreamChunk, OllamaTransportError> {
    serde_json::from_str(line.trim())
        .map_err(|error| OllamaTransportError::MalformedStream(sanitize_detail(&error.to_string())))
}

fn sanitize_detail(value: &str) -> String {
    let normalized = value.replace("authorization", "authentication");
    let end = if normalized.len() <= 4_000 {
        normalized.len()
    } else {
        let mut end = 4_000;
        while !normalized.is_char_boundary(end) {
            end -= 1;
        }
        end
    };
    normalized[..end].to_string()
}

fn sanitize_detail_with_secret(value: &str, secret: Option<&str>) -> String {
    let value = match secret.filter(|secret| !secret.is_empty()) {
        Some(secret) => value.replace(secret, "[redacted]"),
        None => value.to_string(),
    };
    sanitize_detail(&value)
}

#[cfg(test)]
mod tests {
    use super::{
        normalize_base_url, parse_ndjson_line, sanitize_detail, sanitize_detail_with_secret,
        validate_model_tag, OllamaChatMessage, OllamaChatRequest, OllamaClient,
        OllamaTransportError,
    };
    use tokio::sync::oneshot;

    #[test]
    fn normalizes_api_suffix_and_trailing_slashes() {
        assert_eq!(
            normalize_base_url("http://127.0.0.1:11434/api/").unwrap(),
            "http://127.0.0.1:11434"
        );
    }

    #[test]
    fn rejects_remote_plain_http_and_embedded_credentials() {
        assert!(normalize_base_url("http://ollama.example.test").is_err());
        assert!(normalize_base_url("https://user:secret@example.test").is_err());
    }

    #[test]
    fn validates_exact_model_tags_without_rewriting_them() {
        assert!(validate_model_tag("library/llama3.2:8b").is_ok());
        assert!(validate_model_tag("llama3.2:8b ").is_ok());
        assert!(validate_model_tag("llama 3").is_err());
        assert!(validate_model_tag("llama?3").is_err());
    }

    #[test]
    fn parses_deltas_done_markers_and_api_errors() {
        let parsed =
            parse_ndjson_line(r#"{"message":{"role":"assistant","content":"hello"},"done":false}"#)
                .unwrap();
        assert_eq!(parsed.message.unwrap().content, "hello");
        assert!(!parsed.done);
        assert_eq!(
            parse_ndjson_line(r#"{"error":"model not found"}"#)
                .unwrap()
                .error
                .as_deref(),
            Some("model not found")
        );
    }

    #[test]
    fn rejects_malformed_ndjson() {
        assert!(parse_ndjson_line("not-json").is_err());
    }

    #[test]
    fn redacts_remote_bearer_tokens_from_error_details() {
        let detail =
            sanitize_detail_with_secret("remote returned token=super-secret", Some("super-secret"));
        assert!(!detail.contains("super-secret"));
        assert!(detail.contains("[redacted]"));
    }

    #[test]
    fn truncates_error_details_at_a_utf8_boundary() {
        let ascii = sanitize_detail(&"x".repeat(4_100));
        assert_eq!(ascii.len(), 4_000);

        let unicode = sanitize_detail(&"界".repeat(2_000));
        assert!(unicode.len() <= 4_000);
        assert!(unicode.is_char_boundary(unicode.len()));
        assert!(unicode.chars().count() < 2_000);
    }

    #[tokio::test]
    async fn cancelled_completion_stops_before_transport_work() {
        let client =
            OllamaClient::new("http://127.0.0.1:11434", None).expect("client should be valid");
        let request = OllamaChatRequest {
            model: "llama3.2:latest".into(),
            messages: vec![OllamaChatMessage::text("user", "hello")],
            stream: false,
            format: None,
        };
        let (cancel_tx, mut cancel_rx) = oneshot::channel();
        drop(cancel_tx);

        let result = client.complete_chat(&request, &mut cancel_rx).await;

        assert!(matches!(
            result,
            Err(OllamaTransportError::Cancelled)
        ));
    }
}
