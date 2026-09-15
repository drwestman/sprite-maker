use super::{
    image_providers::StoredImageProvider,
    ollama_tools::{self, OllamaToolContext},
};
use crate::{
    conversations::{update_message, update_message_metadata_inner},
    error::{CommandError, CommandResult},
    models::GenerationOptions,
    ollama::{self, OllamaModel},
    ollama_logging,
    ollama_transport::{
        OllamaChatMessage, OllamaChatRequest, OllamaChatResponse, OllamaClient, OllamaToolCall,
    },
    workspace::workspace_path,
    AppState,
};
use serde_json::{json, Value};
use tauri::AppHandle;
use tokio::sync::oneshot;

const MAX_TOOL_TURNS: usize = 32;

pub(crate) struct OllamaAgentRequest<'a> {
    pub app: Option<&'a AppHandle>,
    pub state: &'a AppState,
    pub request_id: &'a str,
    pub conversation_id: &'a str,
    pub workspace_id: &'a str,
    pub worktree_id: Option<&'a str>,
    pub assistant_id: &'a str,
    pub prompt: &'a str,
    pub context: &'a str,
    pub reference_paths: &'a [String],
    pub model: &'a OllamaModel,
    pub model_client: &'a OllamaClient,
    pub reasoning_effort: Option<&'a str>,
    pub generation: Option<&'a GenerationOptions>,
    pub image_provider: Option<&'a StoredImageProvider>,
    pub mflux_generation: Option<&'a crate::mflux::MfluxGenerationRequest>,
    pub image_prompt: &'a str,
    pub command: Option<&'a str>,
}

pub(crate) async fn run(
    request: OllamaAgentRequest<'_>,
    cancel_rx: &mut oneshot::Receiver<()>,
) -> CommandResult<String> {
    let workspace = workspace_path(request.state, request.workspace_id)?;
    tracing::debug!(
        target: "ollama",
        event = "agent_started",
        request_id = %request.request_id,
        conversation_id = %request.conversation_id,
        workspace_id = %request.workspace_id,
        model = %request.model.model,
        native_tools = request.model.tool_calling,
        structured_output = request.model.structured_output,
        reference_count = request.reference_paths.len(),
        "Ollama workspace agent started"
    );
    let mut messages = vec![OllamaChatMessage::text(
        "system",
        system_prompt(
            request.context,
            request.command,
            request.reasoning_effort,
            request.generation,
        ),
    )];
    messages.extend(ollama::load_history(
        request.state,
        request.conversation_id,
    )?);
    ollama::set_latest_user_content(
        &mut messages,
        ollama::format_prompt_for_agent(request.prompt, request.context),
    )?;
    ollama::attach_images(&mut messages, request.reference_paths)?;

    let tools = ollama_tools::definitions();
    let native_tools = request.model.tool_calling;
    let tool_definitions = native_tools.then_some(tools);
    let fallback_format = (!native_tools && request.model.structured_output).then(|| json!("json"));
    let context = OllamaToolContext {
        state: request.state,
        workspace_id: request.workspace_id,
        worktree_id: request.worktree_id,
        workspace: &workspace,
        generation: request.generation,
        image_provider: request.image_provider,
        mflux_generation: request.mflux_generation,
        image_prompt: request.image_prompt,
        command: request.command,
    };
    let mut last_response = String::new();

    for turn in 0..MAX_TOOL_TURNS {
        tracing::debug!(
            target: "ollama",
            event = "agent_turn_started",
            request_id = %request.request_id,
            turn = turn + 1,
            message_count = messages.len(),
            tool_mode = if native_tools { "native" } else { "structured" },
            "Ollama agent turn started"
        );
        let chat_request = OllamaChatRequest {
            model: request.model.model.clone(),
            messages: messages.clone(),
            stream: true,
            format: fallback_format.clone(),
            tools: tool_definitions.clone(),
            think: ollama::thinking_value(request.model, request.reasoning_effort),
        };
        let response = request
            .model_client
            .stream_chat_with_events(&chat_request, cancel_rx, |delta| {
                last_response.push_str(delta);
                let _ = update_message(
                    request.state,
                    request.assistant_id,
                    &last_response,
                    "running",
                );
                emit_agent_event(
                    request.app,
                    request.state,
                    request.request_id,
                    request.conversation_id,
                    "content",
                    delta,
                );
            })
            .await
            .map_err(ollama::transport_error)?;
        tracing::debug!(
            target: "ollama",
            event = "agent_turn_completed",
            request_id = %request.request_id,
            turn = turn + 1,
            response = %ollama_logging::redact_response_for_log(
                &response.content,
                Some(&workspace),
                None,
            ),
            thinking_bytes = response.thinking.len(),
            tool_call_count = response.tool_calls.len(),
            done_reason = ?response.done_reason,
            "Ollama agent turn completed"
        );

        let tool_calls = if native_tools {
            response.tool_calls.clone()
        } else {
            compatibility_tool_calls(&response.content)
        };
        if tool_calls.is_empty() {
            if !native_tools {
                validate_compatibility_response(&response.content)?;
            }
            let final_response =
                compatibility_final_response(&response).unwrap_or(response.content);
            if final_response.trim().is_empty() {
                return Err(CommandError::new(
                    "ollama_empty_response",
                    format!(
                        "Ollama completed ({}) without returning a final response",
                        response.done_reason.as_deref().unwrap_or("unknown reason")
                    ),
                ));
            }
            let mut assistant_message =
                OllamaChatMessage::text("assistant", final_response.clone());
            assistant_message.thinking =
                (!response.thinking.is_empty()).then_some(response.thinking.clone());
            messages.push(assistant_message);
            persist_transcript(request.state, request.assistant_id, &messages, turn)?;
            update_message(
                request.state,
                request.assistant_id,
                &final_response,
                "completed",
            )?;
            tracing::debug!(
                target: "ollama",
                event = "agent_completed",
                request_id = %request.request_id,
                turns = turn + 1,
                response = %ollama_logging::redact_response_for_log(
                    &final_response,
                    Some(&workspace),
                    None,
                ),
                "Ollama workspace agent completed"
            );
            emit_agent_event(
                request.app,
                request.state,
                request.request_id,
                request.conversation_id,
                "completed",
                &final_response,
            );
            return Ok(final_response);
        }

        let assistant_message = OllamaChatMessage::assistant_tool_calls(
            response.content.clone(),
            response.thinking.clone(),
            tool_calls.clone(),
        );
        messages.push(assistant_message);
        persist_transcript(request.state, request.assistant_id, &messages, turn)?;
        for call in tool_calls {
            let tool_name = call.function.name.trim();
            if tool_name.is_empty() {
                return Err(CommandError::new(
                    "ollama_tool_call",
                    "Ollama returned a tool call without a function name",
                ));
            }
            emit_agent_event(
                request.app,
                request.state,
                request.request_id,
                request.conversation_id,
                "activity",
                &format!("Ollama is using {tool_name}"),
            );
            tracing::debug!(
                target: "ollama",
                event = "agent_tool_started",
                request_id = %request.request_id,
                turn = turn + 1,
                tool = %tool_name,
                arguments = %ollama_logging::redact_json_for_log(
                    &call.function.arguments,
                    Some(&workspace),
                    None,
                ),
                "Ollama workspace tool started"
            );
            let result =
                ollama_tools::execute(&context, tool_name, &call.function.arguments, cancel_rx)
                    .await;
            let tool_content = match result {
                Ok(value) => {
                    let safe_result = ollama_logging::redact_json_for_log(
                        &value,
                        Some(&workspace),
                        None,
                    );
                    let serialized = bounded_tool_result(value);
                    tracing::debug!(
                        target: "ollama",
                        event = "agent_tool_completed",
                        request_id = %request.request_id,
                        turn = turn + 1,
                        tool = %tool_name,
                        result = %safe_result,
                        "Ollama workspace tool completed"
                    );
                    emit_agent_event(
                        request.app,
                        request.state,
                        request.request_id,
                        request.conversation_id,
                        "activity",
                        &format!("Completed {tool_name}"),
                    );
                    serialized
                }
                Err(error) if error.code == "request_cancelled" => {
                    tracing::debug!(
                        target: "ollama",
                        event = "agent_tool_cancelled",
                        request_id = %request.request_id,
                        turn = turn + 1,
                        tool = %tool_name,
                        "Ollama workspace tool cancelled"
                    );
                    return Err(error);
                }
                Err(error) => {
                    tracing::debug!(
                        target: "ollama",
                        event = "agent_tool_failed",
                        request_id = %request.request_id,
                        turn = turn + 1,
                        tool = %tool_name,
                        error_code = %error.code,
                        error = %ollama_logging::redact_text(
                            &error.message,
                            Some(&workspace),
                            None,
                        ),
                        "Ollama workspace tool failed"
                    );
                    emit_agent_event(
                        request.app,
                        request.state,
                        request.request_id,
                        request.conversation_id,
                        "activity",
                        &format!("{tool_name} failed: {}", error.message),
                    );
                    serde_json::to_string(&json!({
                        "ok": false,
                        "code": error.code,
                        "error": error.message,
                    }))
                    .unwrap_or_else(|_| "{\"ok\":false,\"error\":\"tool failed\"}".into())
                }
            };
            messages.push(OllamaChatMessage::tool_result(
                call.function.name.clone(),
                tool_content,
            ));
            persist_transcript(request.state, request.assistant_id, &messages, turn)?;
        }
        last_response.clear();
    }

    tracing::debug!(
        target: "ollama",
        event = "agent_tool_limit",
        request_id = %request.request_id,
        max_tool_turns = MAX_TOOL_TURNS,
        "Ollama workspace agent reached the tool-turn limit"
    );
    Err(CommandError::new(
        "ollama_tool_limit",
        "Ollama reached the maximum number of tool turns without completing the request",
    ))
}

fn system_prompt(
    context: &str,
    command: Option<&str>,
    reasoning_effort: Option<&str>,
    generation: Option<&GenerationOptions>,
) -> String {
    let command = command.unwrap_or("none");
    let reasoning_effort = reasoning_effort
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("default");
    let generation = generation
        .and_then(|value| serde_json::to_string(value).ok())
        .unwrap_or_else(|| "{}".into());
    format!(
        "You are the Sprite Studio workspace agent. Work directly in the selected workspace and use tools for every file, command, asset, export, or validation action. Never claim that an action succeeded until its tool result confirms success. Keep all paths inside the selected workspace. Continue iterating until the user's request is complete, then provide a concise final response. If a tool fails, inspect the error and recover or explain the blocker. SLASH COMMAND: {command}. REASONING EFFORT: {reasoning_effort}. GENERATION OPTIONS: {generation}. SELECTED CONTEXT: {context}"
    )
}

fn bounded_tool_result(value: Value) -> String {
    const MAX_RESULT_BYTES: usize = 32 * 1024;
    let serialized = serde_json::to_string(&value).unwrap_or_else(|_| "{}".into());
    if serialized.len() <= MAX_RESULT_BYTES {
        return serialized;
    }
    serde_json::to_string(&json!({
        "ok": false,
        "code": "ollama_tool_output",
        "error": "The tool result exceeded the response limit; request a narrower result.",
        "bytes": serialized.len(),
    }))
    .unwrap_or_else(|_| "{\"ok\":false,\"code\":\"ollama_tool_output\"}".into())
}

fn compatibility_tool_calls(content: &str) -> Vec<OllamaToolCall> {
    let Ok(value) = serde_json::from_str::<Value>(content) else {
        return Vec::new();
    };
    let calls = value
        .get("tool_calls")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| {
            value
                .get("tool")
                .or_else(|| value.get("name"))
                .map(|_| vec![value.clone()])
        })
        .unwrap_or_default();
    calls
        .into_iter()
        .filter_map(|call| {
            let object = call.as_object()?;
            let function = object.get("function").and_then(Value::as_object);
            let name = object
                .get("tool")
                .or_else(|| object.get("name"))
                .or_else(|| function.and_then(|value| value.get("name")))
                .and_then(Value::as_str)?
                .to_string();
            let arguments = object
                .get("arguments")
                .or_else(|| object.get("input"))
                .or_else(|| function.and_then(|value| value.get("arguments")))
                .cloned()
                .unwrap_or_else(|| json!({}));
            Some(OllamaToolCall {
                id: object.get("id").and_then(Value::as_str).map(str::to_string),
                kind: object
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("function")
                    .to_string(),
                index: object
                    .get("index")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize),
                function: crate::ollama_transport::OllamaToolCallFunction {
                    index: object
                        .get("function")
                        .and_then(|function| function.get("index"))
                        .and_then(Value::as_u64)
                        .map(|value| value as usize),
                    name,
                    arguments,
                },
            })
        })
        .collect()
}

fn compatibility_final_response(response: &OllamaChatResponse) -> Option<String> {
    let Ok(value) = serde_json::from_str::<Value>(&response.content) else {
        return None;
    };
    value
        .as_str()
        .map(str::to_string)
        .or_else(|| {
            value
                .get("response")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .or_else(|| {
            value
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

fn validate_compatibility_response(content: &str) -> CommandResult<()> {
    let value = serde_json::from_str::<Value>(content).map_err(|error| {
        CommandError::new(
            "ollama_structured_output",
            format!("Ollama returned invalid structured output: {error}"),
        )
    })?;
    if value.is_string()
        || value.get("response").and_then(Value::as_str).is_some()
        || value.get("message").and_then(Value::as_str).is_some()
    {
        return Ok(());
    }
    Err(CommandError::new(
        "ollama_structured_output",
        "Ollama returned a structured object without a final response or valid tool call",
    ))
}

fn persist_transcript(
    state: &AppState,
    assistant_id: &str,
    messages: &[OllamaChatMessage],
    tool_turn: usize,
) -> CommandResult<()> {
    let metadata = json!({
        "provider": "ollama",
        "ollamaToolTurns": tool_turn + 1,
        "ollamaTranscript": messages,
    });
    update_message_metadata_inner(state, assistant_id, metadata)
}

fn emit_agent_event(
    app: Option<&AppHandle>,
    state: &AppState,
    request_id: &str,
    conversation_id: &str,
    event_type: &str,
    content: &str,
) {
    super::stream::emit_with_metadata(
        app,
        state,
        request_id,
        conversation_id,
        event_type,
        content,
        Some("ollama"),
        None,
    );
}

#[cfg(test)]
mod tests {
    use super::{
        compatibility_final_response, compatibility_tool_calls, validate_compatibility_response,
    };
    use crate::ollama_transport::OllamaChatResponse;

    #[test]
    fn parses_structured_compatibility_tool_calls() {
        let calls =
            compatibility_tool_calls(r#"{"tool":"read_file","arguments":{"path":"README.md"}}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments["path"], "README.md");

        let nested = compatibility_tool_calls(
            r#"{"tool_calls":[{"id":"call-1","type":"function","function":{"name":"read_file","arguments":{"path":"src/lib.rs"}}}]}"#,
        );
        assert_eq!(nested.len(), 1);
        assert_eq!(nested[0].id.as_deref(), Some("call-1"));
        assert_eq!(nested[0].function.arguments["path"], "src/lib.rs");
    }

    #[test]
    fn extracts_structured_final_responses() {
        let response = OllamaChatResponse {
            content: r#"{"response":"done"}"#.into(),
            thinking: String::new(),
            tool_calls: Vec::new(),
            done_reason: Some("stop".into()),
        };
        assert_eq!(
            compatibility_final_response(&response).as_deref(),
            Some("done")
        );
        validate_compatibility_response(r#"{"response":"done"}"#)
            .expect("structured final response");
        assert!(validate_compatibility_response(r#"{"tool_calls":[]}"#).is_err());
    }
}
