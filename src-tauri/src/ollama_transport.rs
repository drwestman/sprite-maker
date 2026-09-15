use base64::Engine;
use reqwest::{Client, Response, StatusCode, Url};
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    time::{Duration, Instant},
};
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(rename = "tool_calls", skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(rename = "tool_name", skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

impl OllamaChatMessage {
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            thinking: None,
            images: None,
            tool_calls: None,
            tool_name: None,
        }
    }

    pub fn assistant_tool_calls(
        content: impl Into<String>,
        thinking: impl Into<String>,
        tool_calls: Vec<OllamaToolCall>,
    ) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
            thinking: Some(thinking.into()).filter(|value| !value.is_empty()),
            images: None,
            tool_calls: Some(tool_calls),
            tool_name: None,
        }
    }

    pub fn tool_result(tool_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
            thinking: None,
            images: None,
            tool_calls: None,
            tool_name: Some(tool_name.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default, skip_serializing_if = "String::is_empty")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    pub function: OllamaToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCallFunction {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OllamaToolFunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolFunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OllamaToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaStreamMessage {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub thinking: String,
    #[serde(default)]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaStreamChunk {
    #[serde(default)]
    pub message: Option<OllamaStreamMessage>,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub done_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OllamaChatResponse {
    pub content: String,
    pub thinking: String,
    pub tool_calls: Vec<OllamaToolCall>,
    pub done_reason: Option<String>,
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
            tracing::debug!(
                target: "ollama",
                event = "http_request_failed",
                endpoint = %self.base_url,
                status = %status,
                "Ollama HTTP request failed"
            );
            Err(OllamaTransportError::Http { status, detail })
        }
    }

    pub async fn tags(&self) -> Result<Vec<OllamaTag>, OllamaTransportError> {
        let started = Instant::now();
        tracing::debug!(
            target: "ollama",
            event = "http_request_started",
            operation = "tags",
            endpoint = %self.endpoint("tags"),
            "Ollama tags request started"
        );
        let response = self
            .authenticated(self.client.get(self.endpoint("tags")))
            .timeout(DISCOVERY_TIMEOUT)
            .send()
            .await;
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                tracing::debug!(
                    target: "ollama",
                    event = "http_request_failed",
                    operation = "tags",
                    endpoint = %self.endpoint("tags"),
                    duration_ms = started.elapsed().as_millis() as u64,
                    "Ollama tags request could not reach the endpoint"
                );
                return Err(OllamaTransportError::Request(sanitize_detail(
                    &error.to_string(),
                )));
            }
        };
        tracing::debug!(
            target: "ollama",
            event = "http_response_received",
            operation = "tags",
            status = %response.status(),
            duration_ms = started.elapsed().as_millis() as u64,
            "Ollama tags response received"
        );
        let response = self.check_response(response).await?;
        let models = response
            .json::<OllamaTagsResponse>()
            .await
            .map(|value| value.models)
            .map_err(|error| {
                OllamaTransportError::MalformedStream(sanitize_detail(&error.to_string()))
            })?;
        tracing::debug!(
            target: "ollama",
            event = "http_request_completed",
            operation = "tags",
            model_count = models.len(),
            duration_ms = started.elapsed().as_millis() as u64,
            "Ollama tags request completed"
        );
        Ok(models)
    }

    pub async fn show(&self, model: &str) -> Result<OllamaShowResponse, OllamaTransportError> {
        validate_model_tag(model)?;
        let started = Instant::now();
        tracing::debug!(
            target: "ollama",
            event = "http_request_started",
            operation = "show",
            endpoint = %self.endpoint("show"),
            model = %model,
            "Ollama model detail request started"
        );
        let response = self
            .authenticated(self.client.post(self.endpoint("show")))
            .timeout(DISCOVERY_TIMEOUT)
            .json(&serde_json::json!({ "model": model }))
            .send()
            .await;
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                tracing::debug!(
                    target: "ollama",
                    event = "http_request_failed",
                    operation = "show",
                    model = %model,
                    endpoint = %self.endpoint("show"),
                    duration_ms = started.elapsed().as_millis() as u64,
                    "Ollama model detail request could not reach the endpoint"
                );
                return Err(OllamaTransportError::Request(sanitize_detail(
                    &error.to_string(),
                )));
            }
        };
        tracing::debug!(
            target: "ollama",
            event = "http_response_received",
            operation = "show",
            status = %response.status(),
            model = %model,
            duration_ms = started.elapsed().as_millis() as u64,
            "Ollama model detail response received"
        );
        let response = self.check_response(response).await?;
        let details = response
            .json::<OllamaShowResponse>()
            .await
            .map_err(|error| {
                OllamaTransportError::MalformedStream(sanitize_detail(&error.to_string()))
            })?;
        tracing::debug!(
            target: "ollama",
            event = "http_request_completed",
            operation = "show",
            model = %model,
            capability_count = details.capabilities.len(),
            duration_ms = started.elapsed().as_millis() as u64,
            "Ollama model detail request completed"
        );
        Ok(details)
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
        Ok(self
            .stream_chat_with_events(request, cancel_rx, |delta| on_delta(delta))
            .await?
            .content)
    }

    pub async fn stream_chat_with_events<F>(
        &self,
        request: &OllamaChatRequest,
        cancel_rx: &mut oneshot::Receiver<()>,
        mut on_delta: F,
    ) -> Result<OllamaChatResponse, OllamaTransportError>
    where
        F: FnMut(&str),
    {
        let mut request = request.clone();
        request.stream = true;
        let started = Instant::now();
        tracing::debug!(
            target: "ollama",
            event = "chat_stream_started",
            model = %request.model,
            message_count = request.messages.len(),
            tool_count = request.tools.as_ref().map(|tools| tools.len()).unwrap_or(0),
            structured_output = request.format.is_some(),
            thinking = request.think.is_some(),
            "Ollama chat request started"
        );
        let response = tokio::select! {
            _ = &mut *cancel_rx => {
                tracing::debug!(
                    target: "ollama",
                    event = "chat_stream_cancelled",
                    model = %request.model,
                    duration_ms = started.elapsed().as_millis() as u64,
                    "Ollama chat request cancelled before response"
                );
                return Err(OllamaTransportError::Cancelled);
            },
            result = self.authenticated(self.client.post(self.endpoint("chat")))
                .json(&request)
                .send() => result,
        };
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                tracing::debug!(
                    target: "ollama",
                    event = "http_request_failed",
                    operation = "chat",
                    model = %request.model,
                    duration_ms = started.elapsed().as_millis() as u64,
                    "Ollama chat request could not reach the endpoint"
                );
                return Err(OllamaTransportError::Request(sanitize_detail(
                    &error.to_string(),
                )));
            }
        };
        tracing::debug!(
            target: "ollama",
            event = "http_response_received",
            operation = "chat",
            status = %response.status(),
            model = %request.model,
            duration_ms = started.elapsed().as_millis() as u64,
            "Ollama chat response received"
        );
        let mut response = self.check_response(response).await?;
        let mut pending = Vec::new();
        let mut content = String::new();
        let mut thinking = String::new();
        let mut tool_calls = Vec::new();
        let mut chunk_count = 0;
        let mut saw_done = false;
        let mut done_reason = None;
        loop {
            let chunk = tokio::select! {
                _ = &mut *cancel_rx => {
                    tracing::debug!(
                        target: "ollama",
                        event = "chat_stream_cancelled",
                        model = %request.model,
                        chunk_count,
                        content_bytes = content.len(),
                        duration_ms = started.elapsed().as_millis() as u64,
                        "Ollama chat request cancelled while streaming"
                    );
                    return Err(OllamaTransportError::Cancelled);
                },
                result = response.chunk() => result,
            }
            .map_err(|error| OllamaTransportError::Request(sanitize_detail(&error.to_string())))?;
            let Some(chunk) = chunk else { break };
            chunk_count += 1;
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
                    tracing::debug!(
                        target: "ollama",
                        event = "chat_stream_api_error",
                        model = %request.model,
                        duration_ms = started.elapsed().as_millis() as u64,
                        "Ollama chat stream returned an API error"
                    );
                    return Err(OllamaTransportError::Api(sanitize_detail_with_secret(
                        &error,
                        self.bearer_token.as_deref(),
                    )));
                }
                if let Some(message) = parsed.message {
                    if !message.thinking.is_empty() {
                        thinking.push_str(&message.thinking);
                    }
                    if !message.content.is_empty() {
                        content.push_str(&message.content);
                        on_delta(&message.content);
                    }
                    if let Some(incoming) = message.tool_calls {
                        merge_tool_calls(&mut tool_calls, incoming);
                    }
                }
                saw_done |= parsed.done;
                if parsed.done_reason.is_some() {
                    done_reason = parsed.done_reason;
                }
            }
        }
        if !pending.iter().all(|byte| byte.is_ascii_whitespace()) {
            let line = String::from_utf8(std::mem::take(&mut pending)).map_err(|error| {
                OllamaTransportError::MalformedStream(sanitize_detail(&error.to_string()))
            })?;
            let parsed = parse_ndjson_line(&line)?;
            if let Some(error) = parsed.error {
                tracing::debug!(
                    target: "ollama",
                    event = "chat_stream_api_error",
                    model = %request.model,
                    duration_ms = started.elapsed().as_millis() as u64,
                    "Ollama chat stream returned an API error"
                );
                return Err(OllamaTransportError::Api(sanitize_detail_with_secret(
                    &error,
                    self.bearer_token.as_deref(),
                )));
            }
            if let Some(message) = parsed.message {
                if !message.thinking.is_empty() {
                    thinking.push_str(&message.thinking);
                }
                if !message.content.is_empty() {
                    content.push_str(&message.content);
                    on_delta(&message.content);
                }
                if let Some(incoming) = message.tool_calls {
                    merge_tool_calls(&mut tool_calls, incoming);
                }
            }
            saw_done |= parsed.done;
            if parsed.done_reason.is_some() {
                done_reason = parsed.done_reason;
            }
        }
        if !saw_done && content.trim().is_empty() && tool_calls.is_empty() {
            return Err(OllamaTransportError::EmptyResponse);
        }
        if content.trim().is_empty() && tool_calls.is_empty() {
            return Err(OllamaTransportError::EmptyResponse);
        }
        for call in &mut tool_calls {
            call.function.arguments = normalize_tool_arguments(&call.function.arguments);
        }
        tracing::debug!(
            target: "ollama",
            event = "chat_stream_completed",
            model = %request.model,
            chunk_count,
            content_bytes = content.len(),
            thinking_bytes = thinking.len(),
            tool_call_count = tool_calls.len(),
            done = saw_done,
            done_reason = ?done_reason,
            duration_ms = started.elapsed().as_millis() as u64,
            "Ollama chat request completed"
        );
        Ok(OllamaChatResponse {
            content,
            thinking,
            tool_calls,
            done_reason,
        })
    }
}

fn merge_tool_calls(existing: &mut Vec<OllamaToolCall>, incoming: Vec<OllamaToolCall>) {
    for (index, mut call) in incoming.into_iter().enumerate() {
        call.function.arguments = normalize_tool_arguments(&call.function.arguments);
        let previous_index = call
            .id
            .as_deref()
            .and_then(|id| {
                existing
                    .iter()
                    .position(|value| value.id.as_deref() == Some(id))
            })
            .or_else(|| {
                call.index.or(call.function.index).and_then(|index| {
                    existing
                        .iter()
                        .position(|value| {
                            value.index.or(value.function.index) == Some(index)
                        })
                })
            })
            .or_else(|| (index < existing.len()).then_some(index));
        if let Some(previous_index) = previous_index {
            let previous = &mut existing[previous_index];
            previous.function.arguments =
                merge_tool_arguments(&previous.function.arguments, &call.function.arguments);
            if previous.function.name.is_empty() {
                previous.function.name = call.function.name;
            }
        } else {
            existing.push(call);
        }
    }
}

fn merge_tool_arguments(previous: &serde_json::Value, next: &serde_json::Value) -> serde_json::Value {
    match (previous, next) {
        (serde_json::Value::Object(previous), serde_json::Value::Object(next)) => {
            let mut merged = previous.clone();
            for (key, value) in next {
                merged.insert(key.clone(), value.clone());
            }
            serde_json::Value::Object(merged)
        }
        (serde_json::Value::String(previous), serde_json::Value::String(next)) => {
            serde_json::Value::String(format!("{previous}{next}"))
        }
        (_, next) => next.clone(),
    }
}

fn normalize_tool_arguments(value: &serde_json::Value) -> serde_json::Value {
    let serde_json::Value::String(arguments) = value else {
        return value.clone();
    };
    serde_json::from_str(arguments).unwrap_or_else(|_| value.clone())
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
        merge_tool_calls, normalize_base_url, parse_ndjson_line, sanitize_detail,
        sanitize_detail_with_secret, validate_model_tag, OllamaChatMessage, OllamaChatRequest,
        OllamaClient, OllamaToolCall, OllamaToolCallFunction, OllamaTransportError,
    };
    use serde_json::json;
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
    fn uses_ollama_snake_case_for_tool_call_messages() {
        let message = OllamaChatMessage::assistant_tool_calls(
            "",
            "",
            vec![OllamaToolCall {
                id: Some("call-1".into()),
                kind: "function".into(),
                index: Some(0),
                function: OllamaToolCallFunction {
                    index: Some(0),
                    name: "read_file".into(),
                    arguments: json!({"path": "README.md"}),
                },
            }],
        );
        let value = serde_json::to_value(message).unwrap();
        assert!(value.get("tool_calls").is_some());
        assert!(value.get("toolCalls").is_none());
        let result = serde_json::to_value(OllamaChatMessage::tool_result(
            "read_file",
            "{\"ok\":true}",
        ))
        .unwrap();
        assert_eq!(result.get("tool_name").and_then(|value| value.as_str()), Some("read_file"));
        assert!(result.get("tool_call_id").is_none());

        let parsed = parse_ndjson_line(
            r#"{"message":{"role":"assistant","tool_calls":[{"id":"call-1","function":{"index":0,"name":"read_file","arguments":{"path":"README.md"}}}]},"done":true}"#,
        )
        .unwrap();
        let message = parsed.message.unwrap();
        let call = message.tool_calls.unwrap().pop().unwrap();
        assert_eq!(
            call.function.index,
            Some(0)
        );
        assert_eq!(
            call.function.name,
            "read_file"
        );
    }

    #[test]
    fn merges_streamed_tool_arguments_by_call_index_without_collapsing_same_named_tools() {
        let mut calls = Vec::new();
        merge_tool_calls(
            &mut calls,
            vec![
                OllamaToolCall {
                    id: None,
                    kind: "function".into(),
                    index: Some(0),
                    function: OllamaToolCallFunction {
                        index: Some(0),
                        name: "read_file".into(),
                        arguments: json!({"path": "a.txt"}),
                    },
                },
                OllamaToolCall {
                    id: None,
                    kind: "function".into(),
                    index: Some(1),
                    function: OllamaToolCallFunction {
                        index: Some(1),
                        name: "read_file".into(),
                        arguments: json!({"path": "b.txt"}),
                    },
                },
            ],
        );
        merge_tool_calls(
            &mut calls,
            vec![
                OllamaToolCall {
                    id: None,
                    kind: "function".into(),
                    index: Some(0),
                    function: OllamaToolCallFunction {
                        index: Some(0),
                        name: "read_file".into(),
                        arguments: json!({"maxBytes": 10}),
                    },
                },
                OllamaToolCall {
                    id: None,
                    kind: "function".into(),
                    index: Some(1),
                    function: OllamaToolCallFunction {
                        index: Some(1),
                        name: "read_file".into(),
                        arguments: json!({"maxBytes": 20}),
                    },
                },
            ],
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.arguments["path"], "a.txt");
        assert_eq!(calls[0].function.arguments["maxBytes"], 10);
        assert_eq!(calls[1].function.arguments["path"], "b.txt");
        assert_eq!(calls[1].function.arguments["maxBytes"], 20);
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
    async fn cancelled_stream_stops_before_transport_work() {
        let client =
            OllamaClient::new("http://127.0.0.1:11434", None).expect("client should be valid");
        let request = OllamaChatRequest {
            model: "llama3.2:latest".into(),
            messages: vec![OllamaChatMessage::text("user", "hello")],
            stream: false,
            format: None,
            tools: None,
            think: None,
        };
        let (cancel_tx, mut cancel_rx) = oneshot::channel();
        drop(cancel_tx);

        let result = client
            .stream_chat(&request, &mut cancel_rx, |_| {})
            .await;

        assert!(matches!(
            result,
            Err(OllamaTransportError::Cancelled)
        ));
    }
}
