use crate::ollama_transport::OllamaChatMessage;
use serde_json::{Map, Value};
use std::path::Path;

const MAX_TEXT_BYTES: usize = 4_000;
const MAX_JSON_BYTES: usize = 8_000;
const MAX_ARRAY_ITEMS: usize = 32;
const MAX_JSON_DEPTH: usize = 6;

pub(crate) fn message_preview(
    message: &OllamaChatMessage,
    workspace: Option<&Path>,
    secret: Option<&str>,
) -> String {
    redact_text(&message.content, workspace, secret)
}

pub(crate) fn redact_text(value: &str, workspace: Option<&Path>, secret: Option<&str>) -> String {
    let mut value = value.replace('\0', "[nul]");
    if let Some(secret) = secret.filter(|secret| !secret.is_empty()) {
        value = value.replace(secret, "[redacted]");
    }
    if let Some(workspace) = workspace {
        let workspace = workspace.to_string_lossy();
        if !workspace.is_empty() {
            value = value.replace(workspace.as_ref(), "[workspace]");
        }
    }
    value = redact_absolute_paths(&value);
    for prefix in [
        "Bearer ",
        "bearer ",
        "api_key=",
        "api_key: ",
        "apiKey=",
        "apiKey: ",
        "access_token=",
        "access_token: ",
        "token=",
        "token: ",
        "x-api-key=",
        "x-api-key: ",
        "authorization=",
        "authorization: ",
    ] {
        value = redact_prefixed_value(&value, prefix);
    }
    truncate(&value, MAX_TEXT_BYTES)
}

pub(crate) fn redact_json_for_log(
    value: &Value,
    workspace: Option<&Path>,
    secret: Option<&str>,
) -> String {
    let value = redact_json(value, workspace, secret, 0);
    let serialized = serde_json::to_string(&value).unwrap_or_else(|_| "\"[redacted]\"".into());
    truncate(&serialized, MAX_JSON_BYTES)
}

pub(crate) fn redact_response_for_log(
    value: &str,
    workspace: Option<&Path>,
    secret: Option<&str>,
) -> String {
    serde_json::from_str::<Value>(value)
        .map(|value| redact_json_for_log(&value, workspace, secret))
        .unwrap_or_else(|_| redact_text(value, workspace, secret))
}

pub(crate) fn safe_path(value: &str, workspace: Option<&Path>) -> String {
    let path = Path::new(value);
    if path.is_absolute() {
        if let Some(workspace) = workspace {
            if let Ok(relative) = path.strip_prefix(workspace) {
                return format!("[workspace]/{}", relative.display());
            }
        }
        return "[absolute-path]".into();
    }
    redact_text(value, workspace, None)
}

fn redact_json(
    value: &Value,
    workspace: Option<&Path>,
    secret: Option<&str>,
    depth: usize,
) -> Value {
    if depth >= MAX_JSON_DEPTH {
        return Value::String("[nested data redacted]".into());
    }
    match value {
        Value::Object(object) => {
            let mut redacted = Map::new();
            for (key, value) in object.iter().take(MAX_ARRAY_ITEMS) {
                let normalized = key.to_ascii_lowercase();
                let value = if is_secret_key(&normalized) {
                    Value::String("[redacted]".into())
                } else if is_content_key(&normalized) {
                    Value::String(format!("[content redacted; {} bytes]", value_size(value)))
                } else if is_path_key(&normalized) {
                    value
                        .as_str()
                        .map(|value| safe_path(value, workspace))
                        .map(Value::String)
                        .unwrap_or_else(|| Value::String("[path redacted]".into()))
                } else {
                    redact_json(value, workspace, secret, depth + 1)
                };
                redacted.insert(key.clone(), value);
            }
            if object.len() > MAX_ARRAY_ITEMS {
                redacted.insert("_truncated".into(), Value::Bool(true));
            }
            Value::Object(redacted)
        }
        Value::Array(values) => {
            let mut redacted = values
                .iter()
                .take(MAX_ARRAY_ITEMS)
                .map(|value| redact_json(value, workspace, secret, depth + 1))
                .collect::<Vec<_>>();
            if values.len() > MAX_ARRAY_ITEMS {
                redacted.push(Value::String("[array truncated]".into()));
            }
            Value::Array(redacted)
        }
        Value::String(value) => Value::String(redact_text(value, workspace, secret)),
        Value::Null | Value::Bool(_) | Value::Number(_) => value.clone(),
    }
}

fn is_secret_key(key: &str) -> bool {
    [
        "token",
        "apikey",
        "api_key",
        "authorization",
        "password",
        "secret",
        "credential",
        "cookie",
    ]
    .iter()
    .any(|part| key.contains(part))
}

fn is_content_key(key: &str) -> bool {
    [
        "content",
        "contents",
        "filedata",
        "filecontent",
        "stdout",
        "stderr",
        "image",
        "images",
        "base64",
        "body",
        "oldtext",
        "newtext",
        "snippet",
        "text",
        "match",
        "patch",
        "diff",
    ]
    .iter()
    .any(|part| key.contains(part))
}

fn is_path_key(key: &str) -> bool {
    key == "path"
        || key == "cwd"
        || key.ends_with("path")
        || key.ends_with("_path")
        || key.contains("filepath")
}

fn value_size(value: &Value) -> usize {
    match value {
        Value::String(value) => value.len(),
        _ => serde_json::to_vec(value)
            .map(|value| value.len())
            .unwrap_or(0),
    }
}

fn redact_prefixed_value(value: &str, prefix: &str) -> String {
    let mut result = String::with_capacity(value.len());
    let mut cursor = 0;
    let normalized_prefix = prefix.to_ascii_lowercase();
    while let Some(relative_start) = value[cursor..]
        .to_ascii_lowercase()
        .find(&normalized_prefix)
    {
        let start = cursor + relative_start;
        let secret_start = start + prefix.len();
        result.push_str(&value[cursor..secret_start]);
        let end = value[secret_start..]
            .char_indices()
            .find(|(_, character)| {
                character.is_whitespace() || matches!(character, '"' | '\'' | ',' | '}')
            })
            .map(|(index, _)| secret_start + index)
            .unwrap_or(value.len());
        result.push_str("[redacted]");
        cursor = end;
    }
    result.push_str(&value[cursor..]);
    result
}

fn redact_absolute_paths(value: &str) -> String {
    let mut result = String::with_capacity(value.len());
    let mut cursor = 0;
    let characters = value.char_indices().collect::<Vec<_>>();
    for (index, (position, character)) in characters.iter().enumerate() {
        let is_unix_path = *character == '/'
            && (index == 0
                || characters[index - 1].1.is_whitespace()
                || matches!(
                    characters[index - 1].1,
                    '"' | '\'' | '(' | '[' | '{' | '=' | ':'
                ));
        let is_windows_path = character.is_ascii_alphabetic()
            && characters
                .get(index + 1)
                .is_some_and(|(_, next)| *next == ':')
            && characters
                .get(index + 2)
                .is_some_and(|(_, next)| *next == '\\' || *next == '/');
        if !is_unix_path && !is_windows_path {
            continue;
        }
        let start = *position;
        let end = characters
            .iter()
            .skip(index + if is_windows_path { 3 } else { 1 })
            .find(|(_, character)| {
                character.is_whitespace()
                    || matches!(character, '"' | '\'' | ',' | ';' | ')' | ']' | '}')
            })
            .map(|(position, _)| *position)
            .unwrap_or(value.len());
        result.push_str(&value[cursor..start]);
        result.push_str("[absolute-path]");
        cursor = end;
    }
    result.push_str(&value[cursor..]);
    result
}

fn truncate(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...[truncated]", &value[..end])
}

#[cfg(test)]
mod tests {
    use super::{message_preview, redact_json_for_log, redact_text, safe_path};
    use crate::ollama_transport::OllamaChatMessage;
    use serde_json::json;
    use std::path::Path;

    #[test]
    fn redacts_secrets_and_workspace_paths_from_text() {
        let workspace = Path::new("/tmp/sprite-workspace");
        let value = "Bearer abc123 token=xyz /tmp/sprite-workspace/assets/fox.png";
        let redacted = redact_text(value, Some(workspace), None);
        assert!(!redacted.contains("abc123"));
        assert!(!redacted.contains("xyz"));
        assert!(!redacted.contains("/tmp/sprite-workspace"));
        assert!(redacted.contains("[workspace]"));
    }

    #[test]
    fn redacts_content_and_sensitive_json_fields() {
        let value = json!({
            "path": "/tmp/sprite-workspace/a.txt",
            "content": "private file contents",
            "oldText": "old private file contents",
            "newText": "new private file contents",
            "stdout": "secret command output",
            "matches": [{"path": "a.txt", "text": "matching file contents"}],
            "token": "abc123",
            "ok": true
        });
        let redacted = redact_json_for_log(&value, Some(Path::new("/tmp/sprite-workspace")), None);
        assert!(!redacted.contains("private file contents"));
        assert!(!redacted.contains("old private file contents"));
        assert!(!redacted.contains("new private file contents"));
        assert!(!redacted.contains("secret command output"));
        assert!(!redacted.contains("matching file contents"));
        assert!(!redacted.contains("abc123"));
        assert!(redacted.contains("[workspace]"));
        assert!(redacted.contains("\"ok\":true"));
    }

    #[test]
    fn previews_message_content_with_the_same_redaction_policy() {
        let message = OllamaChatMessage::text("user", "Bearer abc123 /tmp/workspace");
        let preview = message_preview(&message, Some(Path::new("/tmp/workspace")), None);
        assert!(!preview.contains("abc123"));
        assert!(!preview.contains("/tmp/workspace"));
    }

    #[test]
    fn hides_unrelated_absolute_paths() {
        assert_eq!(
            safe_path(
                "/Users/example/private.txt",
                Some(Path::new("/tmp/workspace"))
            ),
            "[absolute-path]"
        );
        let redacted = redact_text(
            "Read /Users/example/private.txt and /tmp/another.txt",
            Some(Path::new("/tmp/workspace")),
            None,
        );
        assert!(!redacted.contains("/Users/example"));
        assert!(!redacted.contains("/tmp/another"));
    }

    #[test]
    fn redacts_case_insensitive_secret_prefixes() {
        let redacted = redact_text(
            "Authorization: Bearer top-secret X-API-Key=another-secret",
            None,
            None,
        );
        assert!(!redacted.contains("top-secret"));
        assert!(!redacted.contains("another-secret"));
    }
}
