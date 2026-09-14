use super::arguments::{provider_arguments, provider_stdin_bytes};
use super::discovery::provider_process_path;
use super::execute_finish::{finish_provider_run, FinishProviderRun};
use super::external_source::generate_external_source;
use super::headless::apply_tokio_headless_flags;
use super::image_providers::StoredImageProvider;
use super::prompt::create_provider_prompt_file;
use super::stream::{append_stream_text, emit, parse_stream_line, provider_display_name};
use crate::{
    conversations::{set_provider_session, update_message, update_message_metadata_inner},
    workspace::workspace_path,
    AppState,
};
use std::{fs, path::PathBuf, process::Stdio, time::Duration};
use tauri::AppHandle;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
    sync::oneshot,
};

pub(crate) struct ProviderRun {
    pub(crate) request_id: String,
    pub(crate) conversation_id: String,
    pub(crate) workspace_id: String,
    pub(crate) session_id: Option<String>,
    pub(crate) assistant_id: String,
    pub(crate) prompt: String,
    pub(crate) model: Option<String>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) reference_paths: Vec<String>,
    pub(crate) executable: PathBuf,
    pub(crate) image_provider: Option<StoredImageProvider>,
    pub(crate) mflux_generation: Option<crate::mflux::MfluxGenerationRequest>,
    pub(crate) image_prompt: String,
    pub(crate) provider_id: String,
}

fn remove_provider_canceller(state: &AppState, request_id: &str) {
    let _ = state
        .cancellers
        .lock()
        .map(|mut values| values.remove(request_id));
}

pub(crate) async fn run_provider(
    app: Option<AppHandle>,
    state: AppState,
    run: ProviderRun,
    mut cancel_rx: oneshot::Receiver<()>,
) {
    let ProviderRun {
        request_id,
        conversation_id,
        workspace_id,
        session_id,
        assistant_id,
        mut prompt,
        model,
        reasoning_effort,
        mut reference_paths,
        executable,
        image_provider,
        mflux_generation,
        image_prompt,
        provider_id,
    } = run;
    emit(
        app.as_ref(),
        &state,
        &request_id,
        &conversation_id,
        "started",
        format!(
            "{} is working in this workspace",
            provider_display_name(&provider_id)
        ),
    );
    let workspace = match workspace_path(&state, &workspace_id) {
        Ok(path) => path,
        Err(error) => {
            let _ = update_message(&state, &assistant_id, &error.message, "failed");
            emit(
                app.as_ref(),
                &state,
                &request_id,
                &conversation_id,
                "failed",
                error.message,
            );
            remove_provider_canceller(&state, &request_id);
            return;
        }
    };
    if let Some(mflux) = mflux_generation {
        emit(
            app.as_ref(),
            &state,
            &request_id,
            &conversation_id,
            "activity",
            if mflux.frame_requests.is_empty() {
                "Generating the source master with MFLUX Z-Image Turbo"
            } else {
                "Generating the master and animation frames with MFLUX Z-Image Turbo"
            },
        );
        let generated = crate::mflux::generate_outputs(&state, &mflux, &mut cancel_rx).await;
        match generated {
            Ok(output) => {
                let mut provenance = mflux.provenance.clone();
                provenance["masterPath"] = serde_json::Value::String(
                    output
                        .master_path
                        .strip_prefix(&workspace)
                        .unwrap_or(&output.master_path)
                        .to_string_lossy()
                        .into_owned(),
                );
                provenance["framePaths"] = serde_json::Value::Array(
                    output
                        .frame_paths
                        .iter()
                        .map(|path| {
                            serde_json::Value::String(
                                path.strip_prefix(&workspace)
                                    .unwrap_or(path)
                                    .to_string_lossy()
                                    .into_owned(),
                            )
                        })
                        .collect(),
                );
                if let Err(error) = update_message_metadata_inner(
                    &state,
                    &assistant_id,
                    serde_json::json!({ "mflux": provenance }),
                ) {
                    let _ = update_message(&state, &assistant_id, &error.message, "failed");
                    emit(
                        app.as_ref(),
                        &state,
                        &request_id,
                        &conversation_id,
                        "failed",
                        error.message,
                    );
                    remove_provider_canceller(&state, &request_id);
                    return;
                }
                let display = output
                    .master_path
                    .strip_prefix(&workspace)
                    .unwrap_or(&output.master_path)
                    .to_string_lossy();
                if output.frame_paths.is_empty() {
                    prompt = format!("{prompt}\n\nMFLUX PROVIDER CONTRACT\nA source image was generated by MFLUX Z-Image Turbo and attached at `{display}`. Treat this exact attached file as the context source master. Do not call ImageGen for the initial source pass. Inspect it, normalize it with the bundled PNG tools, and continue through the routed harness.");
                } else {
                    let frames = output
                        .frame_paths
                        .iter()
                        .enumerate()
                        .map(|(index, path)| {
                            format!(
                                "{}. `{}`",
                                index + 1,
                                path.strip_prefix(&workspace)
                                    .unwrap_or(path)
                                    .to_string_lossy()
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    prompt = format!(
                        "{prompt}\n\nMFLUX ANIMATION FRAME SET CONTRACT\nMFLUX Z-Image Turbo has already generated the exact source master and every ordered animation frame for this request. The source master is `{display}` and the frame files are:\n{frames}\n\nTreat these files as authoritative generated output. Do not call ImageGen, GenerateImage, generate_image, or the native rig renderer. Inspect the frames, preserve their order and identity, normalize them with the bundled PNG tools, copy them into the routed `assets/<category>/` directory, and write `.sprite-studio/last-generation.json` with the copied frame paths and source master before reporting success. Do not replace them with a pose sheet or independently invented frames."
                    );
                }
                let master_path = output.master_path.to_string_lossy().into_owned();
                if !reference_paths.iter().any(|path| path == &master_path) {
                    reference_paths.push(master_path);
                }
                emit(
                    app.as_ref(),
                    &state,
                    &request_id,
                    &conversation_id,
                    "activity",
                    if output.frame_paths.is_empty() {
                        "MFLUX source master saved; starting generation handoff"
                    } else {
                        "MFLUX master and animation frames saved; starting generation handoff"
                    },
                );
            }
            Err(error) if error.code == "request_cancelled" => {
                let _ = update_message(&state, &assistant_id, "Request cancelled", "cancelled");
                emit(
                    app.as_ref(),
                    &state,
                    &request_id,
                    &conversation_id,
                    "cancelled",
                    "Request cancelled",
                );
                remove_provider_canceller(&state, &request_id);
                return;
            }
            Err(error) => {
                let _ = update_message(&state, &assistant_id, &error.message, "failed");
                emit(
                    app.as_ref(),
                    &state,
                    &request_id,
                    &conversation_id,
                    "failed",
                    error.message,
                );
                remove_provider_canceller(&state, &request_id);
                return;
            }
        }
    } else if let Some(provider) = image_provider.as_ref() {
        emit(
            app.as_ref(),
            &state,
            &request_id,
            &conversation_id,
            "activity",
            format!("Generating the source master with {}", provider.name),
        );
        let generation = generate_external_source(&workspace, provider, &image_prompt);
        tokio::pin!(generation);
        let generated = tokio::select! {
            result = &mut generation => result,
            _ = &mut cancel_rx => {
                let _ = update_message(&state, &assistant_id, "Request cancelled", "cancelled");
                emit(app.as_ref(), &state, &request_id, &conversation_id, "cancelled", "Request cancelled");
                state.cancellers.lock().ok().map(|mut values| values.remove(&request_id));
                return;
            }
        };
        match generated {
            Ok(path) => {
                let display = path
                    .strip_prefix(&workspace)
                    .unwrap_or(&path)
                    .to_string_lossy();
                prompt = format!("EXTERNAL SOURCE MASTER\nA source image was generated by {} and attached at `{display}`. Treat this exact attached file as the context source master. Do not call ImageGen for the initial source pass. Inspect it, normalize it with the bundled PNG tools, and continue through the routed harness.\n\n{prompt}", provider.name);
                reference_paths.push(path.to_string_lossy().into_owned());
                emit(
                    app.as_ref(),
                    &state,
                    &request_id,
                    &conversation_id,
                    "activity",
                    format!(
                        "{} source master saved; starting rig planning",
                        provider.name
                    ),
                );
            }
            Err(error) => {
                let _ = update_message(&state, &assistant_id, &error.message, "failed");
                emit(
                    app.as_ref(),
                    &state,
                    &request_id,
                    &conversation_id,
                    "failed",
                    error.message,
                );
                remove_provider_canceller(&state, &request_id);
                return;
            }
        }
    }
    if !reference_paths.is_empty() {
        emit(
            app.as_ref(),
            &state,
            &request_id,
            &conversation_id,
            "activity",
            format!(
                "Visually inspecting {} attached image{} before rig planning",
                reference_paths.len(),
                if reference_paths.len() == 1 { "" } else { "s" }
            ),
        );
    }
    if provider_id != "codex" && !reference_paths.is_empty() {
        let paths = reference_paths
            .iter()
            .map(|path| format!("- `{path}`"))
            .collect::<Vec<_>>()
            .join("\n");
        prompt.push_str(&format!(
            "\n\nATTACHED REFERENCE FILES\nInspect these local files before acting:\n{paths}"
        ));
    }
    let prompt_file = if provider_id == "grok" {
        match create_provider_prompt_file(&workspace, &request_id, &prompt) {
            Ok(path) => Some(path),
            Err(error) => {
                let message = format!("Could not prepare the Grok prompt: {error}");
                let _ = update_message(&state, &assistant_id, &message, "failed");
                emit(
                    app.as_ref(),
                    &state,
                    &request_id,
                    &conversation_id,
                    "failed",
                    message,
                );
                remove_provider_canceller(&state, &request_id);
                return;
            }
        }
    } else {
        None
    };
    let arguments = provider_arguments(
        &provider_id,
        session_id.as_deref(),
        model.as_deref(),
        reasoning_effort.as_deref(),
        &reference_paths,
        prompt_file.as_deref(),
    );
    let mut command = Command::new(executable);
    command
        .args(&arguments)
        .env("PATH", provider_process_path(&provider_id))
        .current_dir(workspace)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    apply_tokio_headless_flags(&mut command);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            if let Some(path) = prompt_file.as_ref() {
                let _ = fs::remove_file(path);
            }
            let message = format!(
                "Could not start {}: {error}",
                provider_display_name(&provider_id)
            );
            let _ = update_message(&state, &assistant_id, &message, "failed");
            emit(
                app.as_ref(),
                &state,
                &request_id,
                &conversation_id,
                "failed",
                message,
            );
            remove_provider_canceller(&state, &request_id);
            return;
        }
    };
    if provider_id != "grok" {
        if let Some(mut stdin) = child.stdin.take() {
            if let Err(error) = stdin
                .write_all(&provider_stdin_bytes(&provider_id, &prompt))
                .await
            {
                if let Some(path) = prompt_file.as_ref() {
                    let _ = fs::remove_file(path);
                }
                let message = format!(
                    "Could not send the prompt to {}: {error}",
                    provider_display_name(&provider_id)
                );
                let _ = update_message(&state, &assistant_id, &message, "failed");
                emit(
                    app.as_ref(),
                    &state,
                    &request_id,
                    &conversation_id,
                    "failed",
                    message,
                );
                remove_provider_canceller(&state, &request_id);
                return;
            }
        }
    }
    let stderr = child.stderr.take();
    let stderr_task = tauri::async_runtime::spawn(async move {
        let mut output = String::new();
        if let Some(stderr) = stderr {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if !output.is_empty() {
                    output.push('\n');
                }
                output.push_str(&line);
            }
        }
        output
    });
    let mut response = String::new();
    let mut cancelled = false;
    if let Some(stdout) = child.stdout.take() {
        let mut lines = BufReader::new(stdout).lines();
        // Codex can spend several minutes planning or running a tool without
        // emitting a JSON line. Silence is not evidence that its process is
        // stuck, so keep the request alive until it exits or the user stops it.
        // A lightweight heartbeat gives the UI useful feedback in that quiet
        // period without pretending that work has completed.
        let mut heartbeat = tokio::time::interval(Duration::from_secs(30));
        heartbeat.tick().await;
        loop {
            tokio::select! {
                _ = &mut cancel_rx => {
                    cancelled = true;
                    let _ = child.kill().await;
                    break;
                }
                _ = heartbeat.tick() => {
                    emit(app.as_ref(), &state, &request_id, &conversation_id, "activity", format!("{} is still working — you can stop this request at any time", provider_display_name(&provider_id)));
                }
                line = lines.next_line() => match line {
                    Ok(Some(line)) => {
                        let (text, activity, session_id) = parse_stream_line(&provider_id, &line, !response.is_empty());
                        if let Some(text) = text {
                            append_stream_text(&mut response, &provider_id, &text);
                            let _ = update_message(&state, &assistant_id, &response, "running");
                            emit(app.as_ref(), &state, &request_id, &conversation_id, "content", text);
                        }
                        if let Some(activity) = activity { emit(app.as_ref(), &state, &request_id, &conversation_id, "activity", activity); }
                        if let Some(session_id) = session_id {
                            let _ = set_provider_session(&state, &conversation_id, &session_id);
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        emit(app.as_ref(), &state, &request_id, &conversation_id, "activity", format!("Output stream error: {error}"));
                        break;
                    }
                }
            }
        }
    }
    let status = child.wait().await;
    let stderr_output = stderr_task.await.unwrap_or_default();
    if let Some(path) = prompt_file.as_ref() {
        let _ = fs::remove_file(path);
    }
    state
        .cancellers
        .lock()
        .ok()
        .map(|mut values| values.remove(&request_id));
    finish_provider_run(FinishProviderRun {
        app: app.as_ref(),
        state: &state,
        request_id: &request_id,
        conversation_id: &conversation_id,
        workspace_id: &workspace_id,
        assistant_id: &assistant_id,
        provider_id: &provider_id,
        response,
        cancelled,
        status,
        stderr_output: &stderr_output,
    });
}
