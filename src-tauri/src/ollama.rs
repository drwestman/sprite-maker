use crate::{
    error::{CommandError, CommandResult},
    models::{ProviderCapabilities, ProviderConnectionTest, ProviderMode, ProviderStatus},
    ollama_transport::{
        normalize_base_url, validate_model_tag, OllamaChatMessage, OllamaChatRequest, OllamaClient,
        OllamaShowResponse, OllamaTag, OllamaTransportError, DEFAULT_OLLAMA_BASE_URL,
    },
    AppState,
};
use chrono::Utc;
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tauri::State;

const OLLAMA_PROVIDER_KEY: &str = "ollama";
const OLLAMA_MAX_REFERENCES: usize = 10;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OllamaSettingsInput {
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub bearer_token: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OllamaSettings {
    pub base_url: String,
    pub model: Option<String>,
    pub has_token: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OllamaModel {
    pub name: String,
    pub model: String,
    pub modified_at: Option<String>,
    pub size: Option<u64>,
    pub digest: Option<String>,
    pub capabilities: Vec<String>,
    pub family: Option<String>,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
    pub vision: bool,
    pub structured_output: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredOllamaSettings {
    #[serde(default)]
    base_url: String,
    #[serde(default)]
    model: String,
    #[serde(default, alias = "token", alias = "apiKey")]
    bearer_token: String,
}

impl Default for StoredOllamaSettings {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_OLLAMA_BASE_URL.into(),
            model: String::new(),
            bearer_token: String::new(),
        }
    }
}

fn load_settings(state: &AppState) -> CommandResult<StoredOllamaSettings> {
    let connection = state
        .db
        .lock()
        .map_err(|_| CommandError::new("database_locked", "Database lock was poisoned"))?;
    let json: Option<String> = connection
        .query_row(
            "SELECT settings_json FROM provider_settings WHERE provider=?1 AND enabled=1",
            [OLLAMA_PROVIDER_KEY],
            |row| row.get(0),
        )
        .optional()?;
    let mut settings = json
        .map(|value| {
            serde_json::from_str::<StoredOllamaSettings>(&value)
                .map_err(|error| CommandError::new("invalid_ollama_settings", error.to_string()))
        })
        .transpose()?
        .unwrap_or_default();
    if settings.base_url.trim().is_empty() {
        settings.base_url = DEFAULT_OLLAMA_BASE_URL.to_string();
    }
    settings.base_url = normalize_base_url(&settings.base_url)
        .map_err(|error| command_error("invalid_ollama_endpoint", error))?;
    settings.model = settings.model.trim().to_string();
    if !settings.model.is_empty() {
        validate_model_tag(&settings.model)
            .map_err(|error| command_error("invalid_ollama_model", error))?;
    }
    Ok(settings)
}

fn public_settings(settings: &StoredOllamaSettings) -> OllamaSettings {
    OllamaSettings {
        base_url: settings.base_url.clone(),
        model: (!settings.model.is_empty()).then(|| settings.model.clone()),
        has_token: !settings.bearer_token.is_empty(),
    }
}

fn client(settings: &StoredOllamaSettings) -> CommandResult<OllamaClient> {
    OllamaClient::new(
        &settings.base_url,
        (!settings.bearer_token.is_empty()).then_some(settings.bearer_token.as_str()),
    )
    .map_err(|error| command_error("invalid_ollama_endpoint", error))
}

fn command_error(code: &str, error: impl std::fmt::Display) -> CommandError {
    CommandError::new(code, error.to_string())
}

fn model_from_tag(tag: OllamaTag, details: Option<OllamaShowResponse>) -> OllamaModel {
    let capabilities = details
        .as_ref()
        .map(|value| value.capabilities.clone())
        .unwrap_or_default();
    let model_details = details
        .as_ref()
        .and_then(|value| value.details.clone())
        .or(tag.details.clone())
        .unwrap_or_default();
    let vision = capabilities.iter().any(|value| value == "vision");
    // Ollama reports text-generating models as `completion`. Those models can
    // use the native `format` field for JSON/structured output even though
    // the show endpoint does not expose a separate structured-output token.
    let structured_output = capabilities
        .iter()
        .any(|value| value == "completion" || value == "structured_output");
    OllamaModel {
        name: tag.name.clone(),
        model: tag.model.unwrap_or_else(|| tag.name.clone()),
        modified_at: details
            .as_ref()
            .and_then(|value| value.modified_at.clone())
            .or(tag.modified_at),
        size: tag.size,
        digest: tag.digest,
        capabilities,
        family: model_details.family,
        parameter_size: model_details.parameter_size,
        quantization_level: model_details.quantization_level,
        vision,
        structured_output,
    }
}

async fn discover_models(client: &OllamaClient) -> Result<Vec<OllamaModel>, OllamaTransportError> {
    let tags = client.tags().await?;
    let mut models = Vec::with_capacity(tags.len());
    for tag in tags {
        let model_name = tag.model.as_deref().unwrap_or(&tag.name);
        let details = client.show(model_name).await.ok();
        models.push(model_from_tag(tag, details));
    }
    models.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(models)
}

fn provider_mode(model: &OllamaModel) -> ProviderMode {
    let mut capabilities = Vec::new();
    if model.vision {
        capabilities.push("vision");
    }
    if model.structured_output {
        capabilities.push("structured output");
    }
    let description = if capabilities.is_empty() {
        "Ollama model".to_string()
    } else {
        format!("Ollama model · {}", capabilities.join(" · "))
    };
    ProviderMode {
        id: model.model.clone(),
        label: model.name.clone(),
        description,
        default_reasoning_effort: String::new(),
        reasoning_efforts: Vec::new(),
    }
}

fn capabilities(model: Option<&OllamaModel>) -> ProviderCapabilities {
    let vision = model.is_some_and(|value| value.vision);
    let structured_output = model.is_some_and(|value| value.structured_output);
    ProviderCapabilities {
        text_input: true,
        image_input: vision,
        multiple_image_input: vision,
        image_editing: false,
        masks: false,
        transparency: false,
        structured_output,
        video_animation: false,
        image_to_image: false,
        maximum_reference_images: if vision {
            OLLAMA_MAX_REFERENCES as u32
        } else {
            0
        },
    }
}

pub(crate) async fn detect_status(state: &AppState) -> ProviderStatus {
    let settings = match load_settings(state) {
        Ok(value) => value,
        Err(error) => {
            return ProviderStatus {
                id: "ollama".into(),
                name: "Ollama".into(),
                kind: "agent".into(),
                installed: false,
                executable: None,
                status: "needs_setup".into(),
                detail: error.message,
                modes: Vec::new(),
                capabilities: capabilities(None),
                configurable: true,
                has_api_key: false,
                base_url: None,
                model: None,
            };
        }
    };
    let base_url = Some(settings.base_url.clone());
    let selected_model = (!settings.model.is_empty()).then(|| settings.model.clone());
    let client = match client(&settings) {
        Ok(value) => value,
        Err(error) => {
            return ollama_status(
                settings,
                Vec::new(),
                "needs_setup",
                error.message,
                false,
                base_url,
            );
        }
    };
    match discover_models(&client).await {
        Ok(mut models) => {
            let selected = selected_model.as_deref().and_then(|name| {
                models
                    .iter()
                    .find(|model| model.model == name || model.name == name)
            });
            if selected.is_none() {
                if let Some(name) = selected_model.as_deref() {
                    if let Ok(details) = client.show(name).await {
                        let tag = OllamaTag {
                            name: name.into(),
                            model: Some(name.into()),
                            modified_at: None,
                            size: None,
                            digest: None,
                            details: None,
                        };
                        models.push(model_from_tag(tag, Some(details)));
                    }
                }
            }
            let selected = selected_model.as_deref().and_then(|name| {
                models
                    .iter()
                    .find(|model| model.model == name || model.name == name)
            });
            let (status, detail, installed) = match (selected_model.as_deref(), selected) {
                (None, _) => (
                    "needs_setup",
                    "Ollama is reachable. Choose an existing model tag in Settings.".into(),
                    true,
                ),
                (Some(_name), Some(model)) => (
                    "ready",
                    format!(
                        "Connected to Ollama at {} using {}",
                        settings.base_url, model.name
                    ),
                    true,
                ),
                (Some(name), None) => (
                    "needs_setup",
                    format!("Model tag `{name}` is not installed at this Ollama endpoint."),
                    true,
                ),
            };
            ollama_status(settings, models, status, detail, installed, base_url)
        }
        Err(error) => ollama_status(
            settings,
            Vec::new(),
            "offline",
            format!(
                "Ollama is offline at {}. Start Ollama or update the endpoint in Settings: {}",
                client.base_url(),
                error
            ),
            false,
            base_url,
        ),
    }
}

pub(crate) fn detect_status_sync(state: &AppState) -> ProviderStatus {
    let state = state.clone();
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map(|runtime| runtime.block_on(detect_status(&state)))
            .unwrap_or_else(|error| ProviderStatus {
                id: "ollama".into(),
                name: "Ollama".into(),
                kind: "agent".into(),
                installed: false,
                executable: None,
                status: "offline".into(),
                detail: format!("Could not probe Ollama: {error}"),
                modes: Vec::new(),
                capabilities: capabilities(None),
                configurable: true,
                has_api_key: false,
                base_url: None,
                model: None,
            })
    })
    .join()
    .unwrap_or_else(|_| ProviderStatus {
        id: "ollama".into(),
        name: "Ollama".into(),
        kind: "agent".into(),
        installed: false,
        executable: None,
        status: "offline".into(),
        detail: "Could not probe Ollama".into(),
        modes: Vec::new(),
        capabilities: capabilities(None),
        configurable: true,
        has_api_key: false,
        base_url: None,
        model: None,
    })
}

fn ollama_status(
    settings: StoredOllamaSettings,
    models: Vec<OllamaModel>,
    status: &str,
    detail: String,
    installed: bool,
    base_url: Option<String>,
) -> ProviderStatus {
    let selected = if settings.model.is_empty() {
        None
    } else {
        models
            .iter()
            .find(|model| model.model == settings.model || model.name == settings.model)
    };
    let modes = models.iter().map(provider_mode).collect();
    ProviderStatus {
        id: "ollama".into(),
        name: "Ollama".into(),
        kind: "agent".into(),
        installed,
        executable: None,
        status: status.into(),
        detail,
        modes,
        capabilities: capabilities(selected),
        configurable: true,
        has_api_key: !settings.bearer_token.is_empty(),
        base_url,
        model: (!settings.model.is_empty()).then_some(settings.model),
    }
}

pub(crate) async fn selected_model(
    state: &AppState,
    model_override: Option<&str>,
) -> CommandResult<(OllamaClient, OllamaModel)> {
    let settings = load_settings(state)?;
    let model_name = model_override
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(settings.model.as_str());
    if model_name.is_empty() {
        return Err(CommandError::new(
            "ollama_model_required",
            "Choose an existing Ollama model tag in Settings or the provider menu before sending a request",
        ));
    }
    validate_model_tag(model_name).map_err(|error| command_error("invalid_ollama_model", error))?;
    let client = client(&settings)?;
    let details = client
        .show(model_name)
        .await
        .map_err(|error| command_error("ollama_model_unavailable", error))?;
    let model = model_from_tag(
        OllamaTag {
            name: model_name.into(),
            model: Some(model_name.into()),
            modified_at: None,
            size: None,
            digest: None,
            details: None,
        },
        Some(details),
    );
    Ok((client, model))
}

pub(crate) fn validate_reference_images(
    model: &OllamaModel,
    reference_count: usize,
) -> CommandResult<()> {
    if reference_count > 0 && !model.vision {
        return Err(CommandError::new(
            "ollama_vision_required",
            format!(
                "Ollama model `{}` does not support vision; choose a model with the vision capability before attaching references",
                model.model
            ),
        ));
    }
    Ok(())
}

pub(crate) fn request(
    model: &OllamaModel,
    messages: Vec<OllamaChatMessage>,
    format: Option<serde_json::Value>,
) -> OllamaChatRequest {
    OllamaChatRequest {
        model: model.model.clone(),
        messages,
        stream: true,
        format,
    }
}

pub(crate) fn load_history(
    state: &AppState,
    conversation_id: &str,
) -> CommandResult<Vec<OllamaChatMessage>> {
    let connection = state
        .db
        .lock()
        .map_err(|_| CommandError::new("database_locked", "Database lock was poisoned"))?;
    let mut statement = connection.prepare(
        "SELECT role, content FROM messages WHERE conversation_id=?1 AND status='completed' ORDER BY created_at, rowid",
    )?;
    let rows = statement.query_map([conversation_id], |row| {
        Ok(OllamaChatMessage::text(
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
        ))
    })?;
    Ok(rows
        .filter_map(Result::ok)
        .filter(|message| !message.content.trim().is_empty())
        .filter(|message| matches!(message.role.as_str(), "user" | "assistant" | "system"))
        .collect())
}

pub(crate) fn set_latest_user_content(
    messages: &mut [OllamaChatMessage],
    content: impl Into<String>,
) -> CommandResult<()> {
    let Some(message) = messages
        .iter_mut()
        .rev()
        .find(|message| message.role == "user")
    else {
        return Err(CommandError::new(
            "ollama_history_error",
            "The Ollama request has no current user message",
        ));
    };
    message.content = content.into();
    Ok(())
}

pub(crate) fn attach_images(
    messages: &mut [OllamaChatMessage],
    paths: &[String],
) -> CommandResult<()> {
    if paths.is_empty() {
        return Ok(());
    }
    let Some(message) = messages
        .iter_mut()
        .rev()
        .find(|message| message.role == "user")
    else {
        return Err(CommandError::new(
            "ollama_history_error",
            "The Ollama request has no current user message for its references",
        ));
    };
    let mut images = Vec::with_capacity(paths.len());
    for path in paths {
        images.push(
            crate::ollama_transport::encode_image(Path::new(path))
                .map_err(|error| command_error("ollama_reference_error", error))?,
        );
    }
    message.images = Some(images);
    Ok(())
}

pub(crate) fn structured_format() -> Option<serde_json::Value> {
    Some(serde_json::Value::String("json".into()))
}

#[cfg(test)]
pub(crate) fn draft_prompt(
    prompt: &str,
    context: &str,
    command: &str,
    generation: Option<&crate::models::GenerationOptions>,
) -> String {
    let profile = generation
        .and_then(|value| serde_json::to_string(value).ok())
        .unwrap_or_else(|| "{}".into());
    format!(
        "You are the planning model inside Sprite Studio. Create one self-contained, execution-ready prompt for the configured workspace agent. The workspace agent is the only component allowed to inspect or modify files. Do not claim that you created, saved, rendered, inspected, or changed anything. Do not return analysis, markdown headings, code fences, or a progress report: return only the final prompt that another agent can execute. Preserve the requested slash command and all concrete visual constraints. Include the generation profile when it affects the result.\n\nSLASH COMMAND: {command}\nGENERATION PROFILE: {profile}\nSELECTED CONTEXT:\n{context}\n\nUSER REQUEST:\n{prompt}"
    )
}

pub(crate) fn transport_error(error: OllamaTransportError) -> CommandError {
    match error {
        OllamaTransportError::Cancelled => {
            CommandError::new("request_cancelled", "Request cancelled")
        }
        OllamaTransportError::InvalidModel(message) => {
            CommandError::new("invalid_ollama_model", message)
        }
        OllamaTransportError::InvalidEndpoint(message) => {
            CommandError::new("invalid_ollama_endpoint", message)
        }
        other => CommandError::new("ollama_error", other.to_string()),
    }
}

#[tauri::command]
pub fn get_ollama_settings(state: State<'_, AppState>) -> CommandResult<OllamaSettings> {
    Ok(public_settings(&load_settings(&state)?))
}

#[tauri::command]
pub fn save_ollama_settings(
    input: OllamaSettingsInput,
    state: State<'_, AppState>,
) -> CommandResult<OllamaSettings> {
    save_ollama_settings_inner(input, &state)
}

fn save_ollama_settings_inner(
    input: OllamaSettingsInput,
    state: &AppState,
) -> CommandResult<OllamaSettings> {
    let base_url = if input.base_url.trim().is_empty() {
        DEFAULT_OLLAMA_BASE_URL.to_string()
    } else {
        input.base_url
    };
    let base_url = normalize_base_url(&base_url)
        .map_err(|error| command_error("invalid_ollama_endpoint", error))?;
    let model = input.model.trim().to_string();
    if !model.is_empty() {
        validate_model_tag(&model).map_err(|error| command_error("invalid_ollama_model", error))?;
    }
    let existing = load_settings(state)?;
    let bearer_token = if input.bearer_token.trim().is_empty() {
        existing.bearer_token
    } else {
        input.bearer_token.trim().to_string()
    };
    let settings = StoredOllamaSettings {
        base_url,
        model,
        bearer_token,
    };
    let connection = state
        .db
        .lock()
        .map_err(|_| CommandError::new("database_locked", "Database lock was poisoned"))?;
    connection.execute(
        "INSERT INTO provider_settings(provider,enabled,settings_json,updated_at) VALUES (?1,1,?2,?3) ON CONFLICT(provider) DO UPDATE SET enabled=1,settings_json=excluded.settings_json,updated_at=excluded.updated_at",
        params![OLLAMA_PROVIDER_KEY, serde_json::to_string(&settings).map_err(|error| CommandError::new("invalid_ollama_settings", error.to_string()))?, Utc::now().to_rfc3339()],
    )?;
    Ok(public_settings(&settings))
}

#[tauri::command]
pub async fn refresh_ollama_models(state: State<'_, AppState>) -> CommandResult<Vec<OllamaModel>> {
    let settings = load_settings(&state)?;
    discover_models(&client(&settings)?)
        .await
        .map_err(|error| command_error("ollama_unavailable", error))
}

#[tauri::command]
pub async fn test_ollama_connection(
    input: Option<OllamaSettingsInput>,
    state: State<'_, AppState>,
) -> CommandResult<ProviderConnectionTest> {
    test_ollama_connection_inner(input, &state).await
}

async fn test_ollama_connection_inner(
    input: Option<OllamaSettingsInput>,
    state: &AppState,
) -> CommandResult<ProviderConnectionTest> {
    let settings = if let Some(input) = input {
        let base_url = if input.base_url.trim().is_empty() {
            DEFAULT_OLLAMA_BASE_URL.to_string()
        } else {
            input.base_url
        };
        let model = input.model.trim().to_string();
        if !model.is_empty() {
            validate_model_tag(&model)
                .map_err(|error| command_error("invalid_ollama_model", error))?;
        }
        let existing = load_settings(state)?;
        StoredOllamaSettings {
            base_url: normalize_base_url(&base_url)
                .map_err(|error| command_error("invalid_ollama_endpoint", error))?,
            model,
            bearer_token: if input.bearer_token.trim().is_empty() {
                existing.bearer_token
            } else {
                input.bearer_token.trim().to_string()
            },
        }
    } else {
        load_settings(state)?
    };
    let client = client(&settings)?;
    let models = discover_models(&client)
        .await
        .map_err(|error| command_error("ollama_unavailable", error))?;
    if !settings.model.is_empty()
        && !models
            .iter()
            .any(|model| model.model == settings.model || model.name == settings.model)
    {
        client.show(&settings.model).await.map_err(|error| {
            CommandError::new(
                "ollama_model_unavailable",
                format!(
                    "Connected to Ollama, but model tag `{}` is not available: {error}",
                    settings.model
                ),
            )
        })?;
    }
    Ok(ProviderConnectionTest {
        ok: true,
        detail: if settings.model.is_empty() {
            format!(
                "Connected to Ollama at {}. Select an existing model tag to chat.",
                settings.base_url
            )
        } else {
            format!(
                "Connected to Ollama at {} using {}",
                settings.base_url, settings.model
            )
        },
    })
}

#[cfg(test)]
mod tests {
    use super::{
        capabilities, draft_prompt, model_from_tag, ollama_status, provider_mode,
        save_ollama_settings_inner, structured_format, validate_reference_images,
        test_ollama_connection_inner, OllamaSettingsInput, StoredOllamaSettings,
    };
    use crate::ollama_transport::{OllamaShowResponse, OllamaTag};
    use crate::{database, AppState};
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };
    use uuid::Uuid;

    #[test]
    fn maps_completion_and_vision_capabilities_to_provider_contract() {
        let model = model_from_tag(
            OllamaTag {
                name: "vision:latest".into(),
                model: None,
                modified_at: None,
                size: None,
                digest: None,
                details: None,
            },
            Some(OllamaShowResponse {
                capabilities: vec!["completion".into(), "vision".into()],
                details: None,
                modified_at: None,
            }),
        );
        let capabilities = capabilities(Some(&model));
        assert!(capabilities.image_input);
        assert!(capabilities.structured_output);
        assert_eq!(capabilities.maximum_reference_images, 10);
        assert!(provider_mode(&model).description.contains("vision"));
    }

    #[test]
    fn text_only_models_cannot_accept_reference_images() {
        let model = model_from_tag(
            OllamaTag {
                name: "text:latest".into(),
                model: None,
                modified_at: None,
                size: None,
                digest: None,
                details: None,
            },
            Some(OllamaShowResponse {
                capabilities: vec!["completion".into()],
                details: None,
                modified_at: None,
            }),
        );
        assert!(!capabilities(Some(&model)).image_input);
        assert_eq!(capabilities(Some(&model)).maximum_reference_images, 0);
    }

    #[test]
    fn generation_drafts_are_execution_prompts_without_file_claims() {
        let draft = draft_prompt(
            "make a four-frame fire effect",
            "Active project section: effects",
            "effect",
            None,
        );
        assert!(draft.contains("SLASH COMMAND: effect"));
        assert!(draft.contains("return only the final prompt"));
        assert!(draft.contains("Do not claim that you created"));
        assert_eq!(
            structured_format(),
            Some(serde_json::Value::String("json".into()))
        );
    }

    #[test]
    fn rejects_reference_images_for_text_only_models_before_request_setup() {
        let model = model_from_tag(
            OllamaTag {
                name: "text:latest".into(),
                model: None,
                modified_at: None,
                size: None,
                digest: None,
                details: None,
            },
            Some(OllamaShowResponse {
                capabilities: vec!["completion".into()],
                details: None,
                modified_at: None,
            }),
        );

        let error = validate_reference_images(&model, 1).expect_err("vision should be required");
        assert_eq!(error.code, "ollama_vision_required");
        assert!(validate_reference_images(&model, 0).is_ok());
    }

    #[test]
    fn provider_status_exposes_all_discovered_models_not_only_the_default() {
        let settings = StoredOllamaSettings {
            base_url: "http://127.0.0.1:11434".into(),
            model: "vision:latest".into(),
            bearer_token: String::new(),
        };
        let models = vec![
            model_from_tag(
                OllamaTag {
                    name: "text:latest".into(),
                    model: None,
                    modified_at: None,
                    size: None,
                    digest: None,
                    details: None,
                },
                Some(OllamaShowResponse {
                    capabilities: vec!["completion".into()],
                    details: None,
                    modified_at: None,
                }),
            ),
            model_from_tag(
                OllamaTag {
                    name: "vision:latest".into(),
                    model: None,
                    modified_at: None,
                    size: None,
                    digest: None,
                    details: None,
                },
                Some(OllamaShowResponse {
                    capabilities: vec!["completion".into(), "vision".into()],
                    details: None,
                    modified_at: None,
                }),
            ),
        ];

        let status = ollama_status(
            settings,
            models,
            "ready",
            "connected".into(),
            true,
            Some("http://127.0.0.1:11434".into()),
        );

        assert_eq!(
            status
                .modes
                .iter()
                .map(|mode| mode.id.as_str())
                .collect::<Vec<_>>(),
            vec!["text:latest", "vision:latest"]
        );
        assert!(status.capabilities.image_input);
        assert_eq!(status.model.as_deref(), Some("vision:latest"));
    }

    #[test]
    fn saves_normalized_ollama_settings_and_preserves_the_secret_boundary() {
        let root =
            std::env::temp_dir().join(format!("sprite-studio-ollama-test-{}", Uuid::new_v4()));
        let connection = database::open(&root.join("app.sqlite3")).expect("database should open");
        let state = AppState::from_connection(connection);

        let settings = save_ollama_settings_inner(
            OllamaSettingsInput {
                base_url: "http://127.0.0.1:11434/api/".into(),
                model: " llama3.2:latest ".into(),
                bearer_token: "secret-token".into(),
            },
            &state,
        )
        .expect("settings should save");

        assert_eq!(settings.base_url, "http://127.0.0.1:11434");
        assert_eq!(settings.model.as_deref(), Some("llama3.2:latest"));
        assert!(settings.has_token);
        let stored: String = state
            .db
            .lock()
            .expect("database lock")
            .query_row(
                "SELECT settings_json FROM provider_settings WHERE provider='ollama'",
                [],
                |row| row.get(0),
            )
            .expect("stored settings should exist");
        assert!(stored.contains("secret-token"));
        assert!(!stored.contains("\"token\""));
    }

    #[tokio::test]
    async fn tests_connection_against_a_reachable_model_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test endpoint should bind");
        let address = listener.local_addr().expect("test endpoint address");
        let server = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().expect("request should arrive");
                let mut request = Vec::new();
                let mut chunk = [0_u8; 512];
                loop {
                    let read = stream.read(&mut chunk).expect("request should read");
                    request.extend_from_slice(&chunk[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8_lossy(&request);
                let body = if request.starts_with("GET /api/tags") {
                    r#"{"models":[{"name":"llama3.2:latest","model":"llama3.2:latest"}]}"#
                } else {
                    r#"{"capabilities":["completion"]}"#
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("response should write");
            }
        });
        let root = std::env::temp_dir().join(format!("sprite-studio-ollama-test-{}", Uuid::new_v4()));
        let connection = database::open(&root.join("app.sqlite3")).expect("database should open");
        let state = AppState::from_connection(connection);

        let result = test_ollama_connection_inner(
            Some(OllamaSettingsInput {
                base_url: format!("http://{address}"),
                model: "llama3.2:latest".into(),
                bearer_token: String::new(),
            }),
            &state,
        )
        .await
        .expect("connection should succeed");

        server.join().expect("test endpoint should stop");
        assert!(result.ok);
        assert!(result.detail.contains("llama3.2:latest"));
    }
}
