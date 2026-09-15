use super::{
    discovery::find_executable,
    execute::{run_provider, ProviderRun},
    image_providers::{
        image_provider_requires_configuration, is_provider_native_image, load_image_provider,
    },
    modes::provider_is_authenticated,
    ollama_agent::{run as run_ollama_agent, OllamaAgentRequest},
    stream::{
        emit_with_metadata, provider_auth_help, provider_display_name,
        response_reports_generation_failure,
    },
};
use crate::{
    conversations::{add_message, add_message_with_metadata, update_message},
    error::{CommandError, CommandResult},
    models::{Conversation, GenerationOptions, ProviderRequestOptions},
    ollama,
    ollama_logging,
    references,
    settings::get_setting_value,
    sprite_harness::studio_prompt,
    workspace::workspace_path,
    AppState,
};
use serde_json::json;
use tauri::AppHandle;
use tokio::sync::oneshot;
use uuid::Uuid;

struct OllamaRun {
    request_id: String,
    conversation_id: String,
    workspace_id: String,
    workspace_root: std::path::PathBuf,
    worktree_id: Option<String>,
    assistant_id: Option<String>,
    prompt: String,
    context: String,
    reference_paths: Vec<String>,
    model_override: Option<String>,
    reasoning_effort: Option<String>,
    command: Option<String>,
    generation: Option<GenerationOptions>,
    fallback_provider: Option<String>,
    fallback_executable: Option<std::path::PathBuf>,
    native_rig_master_only: bool,
    mflux_generation: Option<crate::mflux::MfluxGenerationRequest>,
    image_prompt: String,
}

pub(crate) fn start_ollama_run(
    conversation: Conversation,
    prompt: String,
    context: Option<String>,
    options: ProviderRequestOptions,
    app: Option<AppHandle>,
    state: &AppState,
) -> CommandResult<String> {
    let (reference_context, reference_paths) = references::prompt_context(
        state,
        &conversation.id,
        &options.reference_ids,
        10,
    )?;
    let combined_context = [context.as_deref().unwrap_or(""), reference_context.as_str()]
        .into_iter()
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");

    let image_provider_id = options
        .image_provider_id
        .as_deref()
        .unwrap_or("imagegen")
        .to_string();
    if image_provider_id == "midjourney" {
        return Err(CommandError::new(
            "provider_unsupported",
            "Midjourney does not provide a public API for this integration",
        ));
    }
    let provider_native_image = is_provider_native_image(&image_provider_id, "ollama");
    let mflux_requested = image_provider_id == "mflux";
    let image_provider = if provider_native_image || mflux_requested {
        None
    } else {
        load_image_provider(state, &image_provider_id)?
    };
    let workspace = workspace_path(state, &conversation.workspace_id)?;
    let image_prompt = format!(
        "{}\n\n{}\n\nCreate one clean, centered, motion-ready game-art source master. Use a plain removable background, clear silhouette, and no text, labels, contact sheet, or multiple poses.",
        prompt, combined_context
    );
    let mflux_generation = crate::mflux::build_generation_request(
        state,
        &conversation.id,
        &workspace,
        &options,
        &image_prompt,
    )?;
    let direct_agent = options.command.is_none() || image_provider.is_some() || mflux_generation.is_some();
    let fallback_provider = if options.generation.is_some() && !direct_agent {
        Some(configured_generation_provider(state)?)
    } else {
        None
    };
    let fallback_executable = fallback_provider
        .as_deref()
        .map(|provider_id| {
            let executable = find_executable(provider_id).ok_or_else(|| {
                CommandError::new(
                    "provider_unavailable",
                    format!(
                        "{} was not found. Install its CLI, authenticate it, then retry detection.",
                        provider_display_name(provider_id)
                    ),
                )
            })?;
            if !provider_is_authenticated(provider_id, &executable) {
                return Err(CommandError::new(
                    "provider_unauthenticated",
                    provider_auth_help(provider_id),
                ));
            }
            Ok(executable)
        })
        .transpose()?;
    let fallback_id = fallback_provider.as_deref().unwrap_or("ollama");
    if fallback_provider.is_some()
        && image_provider_requires_configuration(
            &image_provider_id,
            fallback_id,
            image_provider.is_some(),
        )
    {
        return Err(CommandError::new(
            "provider_unavailable",
            "Configure the selected image provider in Settings before generating",
        ));
    }
    add_message(
        state,
        &conversation.id,
        "user",
        "text",
        &prompt,
        "completed",
    )?;
    let assistant = if fallback_provider.is_none() {
        Some(add_message(
            state,
            &conversation.id,
            "assistant",
            "text",
            "",
            "running",
        )?)
    } else {
        None
    };
    let request_id = Uuid::new_v4().to_string();
    let (cancel_tx, cancel_rx) = oneshot::channel();
    state
        .cancellers
        .lock()
        .map_err(|_| {
            CommandError::new("process_error", "Provider process registry is unavailable")
        })?
        .insert(request_id.clone(), cancel_tx);
    state.track_generation(
        &request_id,
        &conversation.id,
        assistant
            .as_ref()
            .map(|message| message.id.as_str())
            .unwrap_or(""),
        &conversation.workspace_id,
    );

    let run = OllamaRun {
        request_id: request_id.clone(),
        conversation_id: conversation.id,
        workspace_id: conversation.workspace_id,
        workspace_root: workspace.clone(),
        worktree_id: conversation.worktree_id,
        assistant_id: assistant.map(|message| message.id),
        prompt,
        context: combined_context,
        reference_paths,
        model_override: options.model,
        reasoning_effort: options.reasoning_effort,
        command: options.command,
        generation: options.generation,
        fallback_provider,
        fallback_executable,
        native_rig_master_only: options.native_rig_master_only,
        mflux_generation,
        image_prompt,
    };
    let task_state = state.clone();
    tauri::async_runtime::spawn(async move {
        run_ollama(app, task_state, run, image_provider, cancel_rx).await;
    });
    Ok(request_id)
}

fn configured_generation_provider(state: &AppState) -> CommandResult<String> {
    let value = get_setting_value(state, "generation-provider")?;
    let provider = value
        .as_str()
        .unwrap_or("codex")
        .trim()
        .to_ascii_lowercase();
    if !matches!(
        provider.as_str(),
        "codex" | "claude" | "gemini" | "grok" | "cursor" | "antigravity"
    ) {
        return Err(CommandError::new(
            "generation_provider_unsupported",
            "Choose a non-Ollama provider that can execute workspace actions",
        ));
    }
    Ok(provider)
}

async fn run_ollama(
    app: Option<AppHandle>,
    state: AppState,
    run: OllamaRun,
    image_provider: Option<super::image_providers::StoredImageProvider>,
    mut cancel_rx: oneshot::Receiver<()>,
) {
    let command = run.command.as_deref().map(|command| {
        ollama_logging::redact_text(command, Some(&run.workspace_root), None)
    });
    tracing::debug!(
        target: "ollama",
        event = "run_started",
        request_id = %run.request_id,
        conversation_id = %run.conversation_id,
        workspace_id = %run.workspace_id,
        reference_count = run.reference_paths.len(),
        fallback_provider = ?run.fallback_provider,
        command = ?command,
        prompt = %ollama_logging::redact_text(
            &run.prompt,
            Some(&run.workspace_root),
            None,
        ),
        context = %ollama_logging::redact_text(
            &run.context,
            Some(&run.workspace_root),
            None,
        ),
        "Ollama request started"
    );
    emit_with_metadata(
        app.as_ref(),
        &state,
        &run.request_id,
        &run.conversation_id,
        "started",
        "Connecting to Ollama…",
        Some("ollama"),
        run.fallback_provider.as_deref(),
    );

    let (client, model) = match ollama::selected_model(&state, run.model_override.as_deref()).await
    {
        Ok((client, model)) => {
            tracing::debug!(
                target: "ollama",
                event = "model_selected",
                request_id = %run.request_id,
                endpoint = %client.base_url(),
                model = %model.model,
                vision = model.vision,
                structured_output = model.structured_output,
                tool_calling = model.tool_calling,
                thinking = model.thinking,
                "Ollama model selected"
            );
            (client, model)
        }
        Err(error) => {
            finish_ollama_failure(&app, &state, &run, error);
            return;
        }
    };
    if let Err(error) = ollama::validate_reference_images(&model, run.reference_paths.len()) {
        finish_ollama_failure(&app, &state, &run, error);
        return;
    }
    if run.fallback_provider.is_none() {
        let response = run_ollama_agent(
            OllamaAgentRequest {
                app: app.as_ref(),
                state: &state,
                request_id: &run.request_id,
                conversation_id: &run.conversation_id,
                workspace_id: &run.workspace_id,
                worktree_id: run.worktree_id.as_deref(),
                assistant_id: run.assistant_id.as_deref().unwrap_or_default(),
                prompt: &run.prompt,
                context: &run.context,
                reference_paths: &run.reference_paths,
                model: &model,
                model_client: &client,
                reasoning_effort: run.reasoning_effort.as_deref(),
                generation: run.generation.as_ref(),
                image_provider: image_provider.as_ref(),
                mflux_generation: run.mflux_generation.as_ref(),
                image_prompt: &run.image_prompt,
                command: run.command.as_deref(),
            },
            &mut cancel_rx,
        )
        .await;
        match response {
            Ok(_) => {
                tracing::debug!(
                    target: "ollama",
                    event = "run_completed",
                    request_id = %run.request_id,
                    mode = "agent",
                    "Ollama agent request completed"
                );
                if let Ok(mut cancellers) = state.cancellers.lock() {
                    cancellers.remove(&run.request_id);
                }
            }
            Err(error) if error.code == "request_cancelled" => {
                finish_ollama_cancelled(&app, &state, &run);
            }
            Err(error) => {
                finish_ollama_failure(&app, &state, &run, error);
            }
        }
        return;
    }
    let response =
        match run_ollama_request(&app, &state, &run, &client, &model, &mut cancel_rx).await {
            Ok(response) => response,
            Err(error) => {
                if error.code == "request_cancelled" {
                    finish_ollama_cancelled(&app, &state, &run);
                } else {
                    finish_ollama_failure(&app, &state, &run, error);
                }
                return;
            }
        };

    let Some(provider_id) = run.fallback_provider.as_deref() else {
        let _ = state
            .cancellers
            .lock()
            .map(|mut values| values.remove(&run.request_id));
        return;
    };
    let Some(executable) = run.fallback_executable.as_ref() else {
        finish_ollama_failure(
            &app,
            &state,
            &run,
            CommandError::new(
                "provider_unavailable",
                "The configured generation provider is unavailable",
            ),
        );
        return;
    };
    if cancel_rx.try_recv().is_ok() {
        finish_ollama_cancelled(&app, &state, &run);
        return;
    }
    let draft = match add_message_with_metadata(
        &state,
        &run.conversation_id,
        "assistant",
        "text",
        &response,
        "completed",
        json!({
            "provider": "ollama",
            "handoffProvider": provider_id,
            "handoffStatus": "draft",
        }),
    ) {
        Ok(message) => message,
        Err(error) => {
            finish_ollama_failure(&app, &state, &run, error);
            return;
        }
    };
    emit_with_metadata(
        app.as_ref(),
        &state,
        &run.request_id,
        &run.conversation_id,
        "draft",
        response.clone(),
        Some("ollama"),
        Some(provider_id),
    );
    let assistant = match add_message_with_metadata(
        &state,
        &run.conversation_id,
        "assistant",
        "text",
        "",
        "running",
        json!({
            "provider": provider_id,
            "handoffFrom": "ollama",
            "handoffStatus": "running",
            "draftMessageId": draft.id,
        }),
    ) {
        Ok(message) => message,
        Err(error) => {
            finish_ollama_failure(&app, &state, &run, error);
            return;
        }
    };
    state.track_generation(
        &run.request_id,
        &run.conversation_id,
        &assistant.id,
        &run.workspace_id,
    );
    emit_with_metadata(
        app.as_ref(),
        &state,
        &run.request_id,
        &run.conversation_id,
        "handoff",
        format!("Handing generation to {}…", provider_display_name(provider_id)),
        Some("ollama"),
        Some(provider_id),
    );
    tracing::debug!(
        target: "ollama",
        event = "provider_handoff_started",
        request_id = %run.request_id,
        from_provider = "ollama",
        to_provider = %provider_id,
        response = %ollama_logging::redact_response_for_log(
            &response,
            Some(&run.workspace_root),
            None,
        ),
        "Ollama provider handoff started"
    );
    let fallback_run = ProviderRun {
        request_id: run.request_id.clone(),
        conversation_id: run.conversation_id.clone(),
        workspace_id: run.workspace_id.clone(),
        session_id: None,
        assistant_id: assistant.id,
        prompt: studio_prompt(
            &draft.content,
            (!run.context.is_empty()).then_some(run.context.as_str()),
            run.generation.as_ref(),
            run.command.as_deref(),
            Some(provider_id),
            run.native_rig_master_only,
        ),
        model: None,
        reasoning_effort: None,
        reference_paths: run.reference_paths.clone(),
        executable: executable.clone(),
        image_provider,
        mflux_generation: run.mflux_generation,
        image_prompt: format!(
            "{}\n\n{}\n\nCreate one clean, centered, motion-ready game-art source master. Use a plain removable background, clear silhouette, and no text, labels, contact sheet, or multiple poses.",
            draft.content, run.context
        ),
        provider_id: provider_id.to_string(),
    };
    run_provider(app, state, fallback_run, cancel_rx).await;
    tracing::debug!(
        target: "ollama",
        event = "provider_handoff_finished",
        request_id = %run.request_id,
        to_provider = %provider_id,
        "Ollama provider handoff finished"
    );
}

async fn run_ollama_request(
    app: &Option<AppHandle>,
    state: &AppState,
    run: &OllamaRun,
    client: &crate::ollama_transport::OllamaClient,
    model: &ollama::OllamaModel,
    cancel_rx: &mut oneshot::Receiver<()>,
) -> CommandResult<String> {
    let mut messages = ollama::load_history(state, &run.conversation_id)?;
    ollama::set_latest_user_content(&mut messages, format_prompt(&run.prompt, &run.context))?;
    ollama::attach_images(&mut messages, &run.reference_paths)?;
    let format = if run.generation.is_some()
        || run
            .command
            .as_deref()
            .is_some_and(|command| command == "rig")
    {
        ollama::structured_format().filter(|_| model.structured_output)
    } else {
        None
    };
    let request = ollama::request(
        model,
        messages,
        format,
        run.reasoning_effort.as_deref(),
    );
    let latest_user = request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| {
            ollama_logging::message_preview(
                message,
                Some(&run.workspace_root),
                None,
            )
        })
        .unwrap_or_default();
    let started = std::time::Instant::now();
    tracing::debug!(
        target: "ollama",
        event = "chat_request_started",
        request_id = %run.request_id,
        model = %request.model,
        message_count = request.messages.len(),
        reference_count = run.reference_paths.len(),
        structured_output = request.format.is_some(),
        thinking = request.think.is_some(),
        latest_user = %latest_user,
        "Ollama chat stream started"
    );
    let mut response = String::new();
    let transport_result = client
        .stream_chat(&request, cancel_rx, |delta| {
            response.push_str(delta);
            if let Some(assistant_id) = run.assistant_id.as_deref() {
                emit_with_metadata(
                    app.as_ref(),
                    state,
                    &run.request_id,
                    &run.conversation_id,
                    "content",
                    delta,
                    Some("ollama"),
                    None,
                );
                let _ = update_message(state, assistant_id, &response, "running");
            }
        })
        .await;
    if let Err(error) = transport_result {
        let error = ollama::transport_error(error);
        tracing::debug!(
            target: "ollama",
            event = "chat_request_failed",
            request_id = %run.request_id,
            duration_ms = started.elapsed().as_millis() as u64,
            error_code = %error.code,
            error = %ollama_logging::redact_text(&error.message, Some(&run.workspace_root), None),
            "Ollama chat stream failed"
        );
        return Err(error);
    }
    tracing::debug!(
        target: "ollama",
        event = "chat_request_completed",
        request_id = %run.request_id,
        duration_ms = started.elapsed().as_millis() as u64,
        response_bytes = response.len(),
        response = %ollama_logging::redact_response_for_log(
            &response,
            Some(&run.workspace_root),
            None,
        ),
        "Ollama chat stream completed"
    );

    if response_reports_generation_failure(&response) {
        if let Some(assistant_id) = run.assistant_id.as_deref() {
            update_message(state, assistant_id, &response, "failed")?;
        }
        return Err(CommandError::new("generation_failed", response));
    }
    if let Some(assistant_id) = run.assistant_id.as_deref() {
        update_message(state, assistant_id, &response, "completed")?;
        emit_with_metadata(
            app.as_ref(),
            state,
            &run.request_id,
            &run.conversation_id,
            "completed",
            response.clone(),
            Some("ollama"),
            None,
        );
    }
    tracing::debug!(
        target: "ollama",
        event = "run_completed",
        request_id = %run.request_id,
        mode = "handoff",
        response_bytes = response.len(),
        "Ollama request completed before provider handoff"
    );
    Ok(response)
}

fn format_prompt(prompt: &str, context: &str) -> String {
    if context.trim().is_empty() {
        prompt.to_string()
    } else {
        format!("{prompt}\n\nSELECTED CONTEXT:\n{context}")
    }
}

fn finish_ollama_failure(
    app: &Option<AppHandle>,
    state: &AppState,
    run: &OllamaRun,
    error: CommandError,
) {
    tracing::debug!(
        target: "ollama",
        event = "run_failed",
        request_id = %run.request_id,
        error_code = %error.code,
        error = %ollama_logging::redact_text(&error.message, Some(&run.workspace_root), None),
        "Ollama request failed"
    );
    if let Some(assistant_id) = run.assistant_id.as_deref() {
        let _ = update_message(state, assistant_id, &error.message, "failed");
    }
    if let Ok(mut cancellers) = state.cancellers.lock() {
        cancellers.remove(&run.request_id);
    }
    emit_with_metadata(
        app.as_ref(),
        state,
        &run.request_id,
        &run.conversation_id,
        "failed",
        error.message,
        Some("ollama"),
        run.fallback_provider.as_deref(),
    );
}

fn finish_ollama_cancelled(app: &Option<AppHandle>, state: &AppState, run: &OllamaRun) {
    tracing::debug!(
        target: "ollama",
        event = "run_cancelled",
        request_id = %run.request_id,
        "Ollama request cancelled"
    );
    if let Some(assistant_id) = run.assistant_id.as_deref() {
        let _ = update_message(state, assistant_id, "Request cancelled", "cancelled");
    }
    if let Ok(mut cancellers) = state.cancellers.lock() {
        cancellers.remove(&run.request_id);
    }
    emit_with_metadata(
        app.as_ref(),
        state,
        &run.request_id,
        &run.conversation_id,
        "cancelled",
        "Request cancelled",
        Some("ollama"),
        run.fallback_provider.as_deref(),
    );
}

#[cfg(test)]
mod tests {
    use super::format_prompt;

    #[test]
    fn appends_context_without_losing_the_user_prompt() {
        assert_eq!(
            format_prompt("make a fox", "Workspace style: flat"),
            "make a fox\n\nSELECTED CONTEXT:\nWorkspace style: flat"
        );
    }
}
