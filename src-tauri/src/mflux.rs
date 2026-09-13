use crate::{
    error::{CommandError, CommandResult},
    models::{
        MfluxSettings, MfluxSettingsInput, MfluxSetupEvent, MotionPlan, ProviderRequestOptions,
    },
    motion_planner::build_motion_plan,
    app_data_dir, AppState,
};
use chrono::Utc;
use rusqlite::OptionalExtension;
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Stdio,
};
use tauri::{AppHandle, Emitter, State};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::oneshot,
};
use uuid::Uuid;

const MFLUX_PROVIDER_KEY: &str = "mflux";
const DEFAULT_REPOSITORY: &str = "mflux-community/z-image-turbo-mflux-q8";
const DEFAULT_REVISION: &str = "4430e72e37bf2bc7bc889a42d306ae1b8d3b22de";
const RUNTIME_VERSION: &str = "python3.11;mflux==0.19.1;mlx==0.32.0";

pub(crate) struct MfluxWorkerProcess {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[derive(Debug, Clone)]
pub(crate) struct MfluxAnimationFrame {
    pub output_path: PathBuf,
    pub prompt: String,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct MfluxGenerationRequest {
    pub repository: String,
    pub revision: String,
    pub output_path: PathBuf,
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f64,
    pub seed: u64,
    pub image_path: Option<PathBuf>,
    pub image_strength: Option<f64>,
    pub source_master_path: Option<PathBuf>,
    pub frame_requests: Vec<MfluxAnimationFrame>,
    pub frame_image_strength: f64,
    pub provenance: serde_json::Value,
}

pub(crate) struct MfluxGeneratedOutput {
    pub master_path: PathBuf,
    pub frame_paths: Vec<PathBuf>,
}

impl MfluxWorkerProcess {
    async fn read_response(&mut self, request_id: &str) -> CommandResult<serde_json::Value> {
        loop {
            let mut line = String::new();
            let bytes = self.stdout.read_line(&mut line).await.map_err(|error| {
                CommandError::new("mflux_worker_failed", format!("Could not read MFLUX worker output: {error}"))
            })?;
            if bytes == 0 {
                return Err(CommandError::new(
                    "mflux_worker_failed",
                    "The MFLUX worker exited before returning a response",
                ));
            }
            let response: serde_json::Value = serde_json::from_str(line.trim()).map_err(|error| {
                CommandError::new(
                    "mflux_worker_failed",
                    format!("The MFLUX worker returned invalid JSON: {error}"),
                )
            })?;
            if response.get("id").and_then(serde_json::Value::as_str) == Some(request_id) {
                return Ok(response);
            }
        }
    }

    async fn ping(&mut self) -> CommandResult<()> {
        let request_id = Uuid::new_v4().to_string();
        let payload = serde_json::json!({ "id": request_id, "type": "ping" });
        let mut bytes = serde_json::to_vec(&payload).map_err(|error| {
            CommandError::new("mflux_worker_failed", format!("Could not encode MFLUX ping: {error}"))
        })?;
        bytes.push(b'\n');
        self.stdin.write_all(&bytes).await.map_err(|error| {
            CommandError::new("mflux_worker_failed", format!("Could not send MFLUX ping: {error}"))
        })?;
        self.stdin.flush().await.map_err(|error| {
            CommandError::new("mflux_worker_failed", format!("Could not flush MFLUX ping: {error}"))
        })?;
        let response = self.read_response(&request_id).await?;
        if response.get("type").and_then(serde_json::Value::as_str) != Some("pong")
            || response.get("runtimeVersion").and_then(serde_json::Value::as_str)
                != Some(RUNTIME_VERSION)
        {
            return Err(CommandError::new(
                "mflux_worker_failed",
                "The MFLUX worker reported an incompatible runtime",
            ));
        }
        Ok(())
    }

    async fn generate(
        &mut self,
        request: &MfluxGenerationRequest,
        cancel_rx: &mut oneshot::Receiver<()>,
    ) -> CommandResult<PathBuf> {
        let request_id = Uuid::new_v4().to_string();
        let mut payload = serde_json::json!({
            "id": request_id,
            "type": "generate",
            "repository": request.repository,
            "revision": request.revision,
            "checkpointPath": cache_root().join("checkpoints"),
            "outputPath": request.output_path,
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "guidance": request.guidance,
            "seed": request.seed,
        });
        if let Some(image_path) = request.image_path.as_ref() {
            payload["imagePath"] = serde_json::Value::String(image_path.to_string_lossy().into_owned());
            payload["imageStrength"] = serde_json::json!(request.image_strength);
        }
        let mut bytes = serde_json::to_vec(&payload).map_err(|error| {
            CommandError::new(
                "mflux_worker_failed",
                format!("Could not encode MFLUX generation request: {error}"),
            )
        })?;
        bytes.push(b'\n');
        self.stdin.write_all(&bytes).await.map_err(|error| {
            CommandError::new(
                "mflux_worker_failed",
                format!("Could not send MFLUX generation request: {error}"),
            )
        })?;
        self.stdin.flush().await.map_err(|error| {
            CommandError::new(
                "mflux_worker_failed",
                format!("Could not flush MFLUX generation request: {error}"),
            )
        })?;

        let mut cancellation_sent = false;
        loop {
            let mut line = String::new();
            let read_result = if cancellation_sent {
                self.stdout.read_line(&mut line).await
            } else {
                tokio::select! {
                    result = self.stdout.read_line(&mut line) => result,
                    _ = &mut *cancel_rx => {
                        let cancel = serde_json::json!({ "id": request_id, "type": "cancel" });
                        let mut cancel_bytes = serde_json::to_vec(&cancel).map_err(|error| {
                            CommandError::new("mflux_worker_failed", format!("Could not encode MFLUX cancellation: {error}"))
                        })?;
                        cancel_bytes.push(b'\n');
                        self.stdin.write_all(&cancel_bytes).await.map_err(|error| {
                            CommandError::new("mflux_worker_failed", format!("Could not send MFLUX cancellation: {error}"))
                        })?;
                        self.stdin.flush().await.map_err(|error| {
                            CommandError::new("mflux_worker_failed", format!("Could not flush MFLUX cancellation: {error}"))
                        })?;
                        cancellation_sent = true;
                        continue;
                    }
                }
            };
            let bytes_read = read_result.map_err(|error| {
                CommandError::new("mflux_worker_failed", format!("Could not read MFLUX worker output: {error}"))
            })?;
            if bytes_read == 0 {
                return Err(CommandError::new(
                    "mflux_worker_failed",
                    "The MFLUX worker exited before generation completed",
                ));
            }
            let response: serde_json::Value = serde_json::from_str(line.trim()).map_err(|error| {
                CommandError::new(
                    "mflux_worker_failed",
                    format!("The MFLUX worker returned invalid JSON: {error}"),
                )
            })?;
            if response.get("id").and_then(serde_json::Value::as_str) != Some(request_id.as_str()) {
                continue;
            }
            match response.get("type").and_then(serde_json::Value::as_str) {
                Some("completed") => {
                    let output = response
                        .get("outputPath")
                        .and_then(serde_json::Value::as_str)
                        .map(PathBuf::from)
                        .ok_or_else(|| {
                            CommandError::new(
                                "mflux_worker_failed",
                                "The MFLUX worker did not return an output path",
                            )
                        })?;
                    if output != request.output_path || !output.is_file() {
                        return Err(CommandError::new(
                            "mflux_worker_failed",
                            "MFLUX returned an invalid output path",
                        ));
                    }
                    return Ok(output);
                }
                Some("cancelled") => {
                    return Err(CommandError::new("request_cancelled", "Request cancelled"));
                }
                Some("error") => {
                    return Err(CommandError::new(
                        "mflux_generation_failed",
                        response
                            .get("error")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("MFLUX generation failed"),
                    ));
                }
                _ => {}
            }
        }
    }
}

async fn spawn_worker() -> CommandResult<MfluxWorkerProcess> {
    let mut child = Command::new(runtime_root().join("bin/python"))
        .arg(worker_path())
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| {
            CommandError::new(
                "mflux_worker_failed",
                format!("Could not start the managed MFLUX worker: {error}"),
            )
        })?;
    let stdin = child.stdin.take().ok_or_else(|| {
        CommandError::new("mflux_worker_failed", "The MFLUX worker did not expose stdin")
    })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        CommandError::new("mflux_worker_failed", "The MFLUX worker did not expose stdout")
    })?;
    if let Some(stderr) = child.stderr.take() {
        tauri::async_runtime::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("[mflux] {line}");
            }
        });
    }
    let mut worker = MfluxWorkerProcess {
        _child: child,
        stdin,
        stdout: BufReader::new(stdout),
    };
    worker.ping().await?;
    Ok(worker)
}

pub(crate) async fn generate_outputs(
    state: &crate::AppState,
    request: &MfluxGenerationRequest,
    cancel_rx: &mut oneshot::Receiver<()>,
) -> CommandResult<MfluxGeneratedOutput> {
    write_managed_resources()?;
    if !runtime_ready() {
        return Err(CommandError::new(
            "mflux_setup_required",
            "Install or repair the managed MFLUX runtime before generating",
        ));
    }
    if let Some(parent) = request.output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(path) = request.source_master_path.as_ref() {
        if !path.is_file() {
            return Err(CommandError::new(
                "mflux_source_missing",
                "The selected MFLUX source master is missing from disk",
            ));
        }
    }
    let mut worker = tokio::select! {
        worker = state.mflux_worker.lock() => worker,
        _ = &mut *cancel_rx => return Err(CommandError::new("request_cancelled", "Request cancelled")),
    };
    if worker.is_none() {
        *worker = Some(spawn_worker().await?);
    }
    let process = worker.as_mut().expect("MFLUX worker is initialized");
    let mut generated_paths = Vec::new();
    let result = async {
        let master_path = if let Some(path) = request.source_master_path.as_ref() {
            path.clone()
        } else {
            generated_paths.push(request.output_path.clone());
            let path = process.generate(request, cancel_rx).await?;
            path
        };
        let mut frame_paths = Vec::with_capacity(request.frame_requests.len());
        for frame in &request.frame_requests {
            let frame_request = MfluxGenerationRequest {
                repository: request.repository.clone(),
                revision: request.revision.clone(),
                output_path: frame.output_path.clone(),
                prompt: frame.prompt.clone(),
                width: request.width,
                height: request.height,
                steps: request.steps,
                guidance: request.guidance,
                seed: frame.seed,
                image_path: Some(master_path.clone()),
                image_strength: Some(request.frame_image_strength),
                source_master_path: None,
                frame_requests: Vec::new(),
                frame_image_strength: request.frame_image_strength,
                provenance: request.provenance.clone(),
            };
            generated_paths.push(frame.output_path.clone());
            let path = process.generate(&frame_request, cancel_rx).await?;
            frame_paths.push(path);
        }
        if let Some(error) = generated_paths
            .iter()
            .find(|path| !path.is_file())
            .map(|path| format!("MFLUX output disappeared: {}", path.display()))
        {
            return Err(CommandError::new("mflux_worker_failed", error));
        }
        Ok(MfluxGeneratedOutput {
            master_path,
            frame_paths,
        })
    }
    .await;
    if result.is_err() {
        cleanup_failed_generation(&mut *worker, generated_paths);
    }
    result
}

fn supported_host() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

fn cache_root() -> PathBuf {
    app_data_dir().join("mflux")
}

fn runtime_root() -> PathBuf {
    cache_root().join("runtime")
}

fn worker_path() -> PathBuf {
    cache_root().join("mflux_worker.py")
}

fn runtime_marker_path() -> PathBuf {
    runtime_root().join(".sprite-studio-mflux-runtime")
}

fn default_input() -> MfluxSettingsInput {
    MfluxSettingsInput {
        repository: DEFAULT_REPOSITORY.into(),
        revision: DEFAULT_REVISION.into(),
    }
}

fn load_input(state: &AppState) -> CommandResult<MfluxSettingsInput> {
    let connection = state
        .db
        .lock()
        .map_err(|_| CommandError::new("database_locked", "Database lock was poisoned"))?;
    let value: Option<String> = connection
        .query_row(
            "SELECT settings_json FROM provider_settings WHERE provider=?1 AND enabled=1",
            [MFLUX_PROVIDER_KEY],
            |row| row.get(0),
        )
        .optional()?;
    value
        .map(|json| {
            serde_json::from_str(&json).map_err(|error| {
                CommandError::new(
                    "invalid_mflux_settings",
                    format!("Stored MFLUX settings are invalid: {error}"),
                )
            })
        })
        .transpose()
        .map(|value| value.unwrap_or_else(default_input))
}

fn validate_repository(value: &str) -> CommandResult<String> {
    let value = value.trim();
    let Some((owner, name)) = value.split_once('/') else {
        return Err(CommandError::new(
            "invalid_mflux_repository",
            "Use a Hugging Face repository in the form owner/name",
        ));
    };
    if owner.is_empty()
        || name.is_empty()
        || matches!(owner, "." | "..")
        || matches!(name, "." | "..")
        || value.len() > 256
        || !owner
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.'))
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.'))
    {
        return Err(CommandError::new(
            "invalid_mflux_repository",
            "Use a Hugging Face repository in the form owner/name",
        ));
    }
    Ok(value.into())
}

fn validate_revision(value: &str) -> CommandResult<String> {
    let value = value.trim();
    if value.len() != 40 || !value.chars().all(|character| character.is_ascii_hexdigit()) {
        return Err(CommandError::new(
            "invalid_mflux_revision",
            "MFLUX revisions must be immutable 40-character commit hashes",
        ));
    }
    Ok(value.into())
}

fn validate_input(input: MfluxSettingsInput) -> CommandResult<MfluxSettingsInput> {
    Ok(MfluxSettingsInput {
        repository: validate_repository(&input.repository)?,
        revision: validate_revision(&input.revision)?,
    })
}

fn runtime_ready() -> bool {
    supported_host()
        && runtime_marker_path().is_file()
        && fs::read_to_string(runtime_marker_path())
            .map(|value| value.trim() == RUNTIME_VERSION)
            .unwrap_or(false)
        && runtime_root().join("bin/python").is_file()
        && worker_path().is_file()
}

fn status_for(input: MfluxSettingsInput) -> MfluxSettings {
    let supported_host = supported_host();
    let runtime_ready = runtime_ready();
    let (status, detail) = if !supported_host {
        (
            "unsupported",
            "MFLUX requires Apple Silicon macOS and is unavailable on this host.",
        )
    } else if runtime_ready {
        (
            "ready",
            "MFLUX Z-Image Turbo is ready. The checkpoint downloads into the managed cache on first generation.",
        )
    } else {
        (
            "needs_setup",
            "Install or repair the managed MFLUX runtime before selecting this provider.",
        )
    };
    MfluxSettings {
        repository: input.repository,
        revision: input.revision,
        runtime_version: RUNTIME_VERSION.into(),
        runtime_ready,
        supported_host,
        status: status.into(),
        detail: detail.into(),
        cache_path: cache_root().to_string_lossy().into_owned(),
    }
}

pub(crate) fn detect_settings(state: &AppState) -> MfluxSettings {
    match load_input(state).and_then(validate_input) {
        Ok(input) => status_for(input),
        Err(error) => {
            let mut settings = status_for(default_input());
            settings.status = "unavailable".into();
            settings.detail = format!("MFLUX settings could not be loaded: {}", error.message);
            settings
        }
    }
}

fn source_size_and_steps(quality: &str) -> (u32, u32) {
    if quality == "low" {
        (512, 4)
    } else {
        (1024, 9)
    }
}

fn prompt_requests_animation(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    [
        "animate",
        "animation",
        "animated",
        "cycle",
        "walking",
        "walk ",
        "running",
        "run ",
        "hopping",
        "hop ",
        "flying",
        "fly ",
        "crawling",
        "crawl ",
        "idle",
        "breathing",
        "attack",
        "slash",
        "strike",
        "cast",
        "explosion",
        "effect",
    ]
    .iter()
    .any(|term| lower.contains(term))
}

fn animation_requested(options: &ProviderRequestOptions, prompt: &str) -> bool {
    options.command.as_deref() == Some("animate")
        || (options.command.is_none() && prompt_requests_animation(prompt))
}

fn validate_reference_selection<'a>(
    selected_id: Option<&'a str>,
    reference_ids: &[String],
    focused_id: Option<&str>,
) -> CommandResult<&'a str> {
    let selected_id = selected_id.ok_or_else(|| {
        CommandError::new(
            "mflux_reference_required",
            "MFLUX image-to-image requires one focused reference, or exactly one active reference",
        )
    })?;
    if !reference_ids.iter().any(|id| id == selected_id) {
        return Err(CommandError::new(
            "mflux_reference_required",
            "The selected MFLUX reference must be active in this conversation",
        ));
    }
    if reference_ids.len() > 1 && focused_id != Some(selected_id) {
        return Err(CommandError::new(
            "mflux_reference_required",
            "Focus one reference before using MFLUX image-to-image with multiple active references",
        ));
    }
    Ok(selected_id)
}

fn generation_allowed(options: &ProviderRequestOptions, prompt: &str) -> bool {
    matches!(
        options.command.as_deref(),
        Some("sprite" | "character" | "effect" | "pack")
    ) || animation_requested(options, prompt)
        || options.native_rig_master_only
}

fn source_master_path(workspace: &Path, value: Option<&str>) -> CommandResult<Option<PathBuf>> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let relative = Path::new(value);
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(component, std::path::Component::ParentDir)
        })
        || !matches!(
            relative.components().next(),
            Some(std::path::Component::Normal(component)) if component == "assets"
        )
    {
        return Err(CommandError::new(
            "invalid_mflux_source",
            "MFLUX animation sources must be workspace-relative files under assets/",
        ));
    }
    let path = workspace.join(relative);
    let assets_root = workspace.join("assets").canonicalize().map_err(|_| {
        CommandError::new(
            "invalid_mflux_source",
            "The selected MFLUX animation source master does not exist",
        )
    })?;
    let path = path.canonicalize().map_err(|_| {
        CommandError::new(
            "invalid_mflux_source",
            "The selected MFLUX animation source master does not exist",
        )
    })?;
    if !path.is_file() || !path.starts_with(&assets_root) {
        return Err(CommandError::new(
            "invalid_mflux_source",
            "MFLUX animation sources must remain inside the workspace assets directory",
        ));
    }
    Ok(Some(path))
}

fn frame_prompts(prompt: &str, plan: &MotionPlan, frame_count: u32) -> Vec<String> {
    let phases = plan
        .phases
        .iter()
        .flat_map(|phase| {
            std::iter::repeat_n(&phase.description, phase.frame_count as usize)
        })
        .take(frame_count as usize)
        .collect::<Vec<_>>();
    (0..frame_count)
        .map(|index| {
            let phase = phases
                .get(index as usize)
                .copied()
                .unwrap_or(&plan.explanation);
            format!(
                "{prompt}\n\nMFLUX ANIMATION FRAME {}/{}\nGenerate exactly one transparent game-art animation frame from the attached source master. Preserve the exact subject identity, silhouette, camera, scale, palette, ground line, and canvas placement. Change only the pose or visible action needed for this frame. Motion phase: {phase}. Do not create a contact sheet, pose sheet, labels, text, or multiple views.",
                index + 1,
                frame_count
            )
        })
        .collect()
}

pub(crate) fn build_generation_request(
    state: &AppState,
    conversation_id: &str,
    workspace: &Path,
    options: &ProviderRequestOptions,
    prompt: &str,
) -> CommandResult<Option<MfluxGenerationRequest>> {
    if options.image_provider_id.as_deref() != Some("mflux") {
        return Ok(None);
    }
    if !generation_allowed(options, prompt) {
        return Err(CommandError::new(
            "mflux_generation_scope",
            "MFLUX is available for static generation and animated image requests; use the chat provider for ordinary chat and /rig turns",
        ));
    }
    let settings = status_for(validate_input(load_input(state)?)?);
    if settings.status != "ready" {
        return Err(CommandError::new(
            "mflux_setup_required",
            settings.detail,
        ));
    }
    let generation = options.generation.as_ref().ok_or_else(|| {
        CommandError::new(
            "mflux_generation_profile_required",
            "MFLUX generation requires a generation profile",
        )
    })?;
    let (source_size, steps) = source_size_and_steps(&generation.quality);
    let requested_animation = animation_requested(options, prompt);
    let source_master_path = if requested_animation {
        source_master_path(workspace, options.source_asset_path.as_deref())?
    } else {
        None
    };
    let selected_reference = if generation.image_input_mode == "image-to-image"
        && source_master_path.is_none()
    {
        let focused_id = if options.reference_ids.len() > 1 {
            crate::settings::get_setting_value(
                state,
                &format!("conversation-focus:{conversation_id}"),
            )?
            .as_str()
            .map(str::to_owned)
        } else {
            None
        };
        let selected_id = validate_reference_selection(
            options.mflux_reference_id.as_deref(),
            &options.reference_ids,
            focused_id.as_deref(),
        )?;
        Some(crate::references::selected_reference_input(
            state,
            conversation_id,
            selected_id,
        )?)
    } else {
        None
    };
    let output_id = Uuid::new_v4();
    let output_path = workspace
        .join(".sprite-studio/provider-sources")
        .join(format!("mflux-{output_id}-master.png"));
    let motion_plan = if requested_animation {
        Some(build_motion_plan(prompt, generation)?)
    } else {
        None
    };
    let frame_count = motion_plan
        .as_ref()
        .map(|plan| plan.selected_frame_count)
        .unwrap_or(0);
    let frame_prompts = motion_plan
        .as_ref()
        .map(|plan| frame_prompts(prompt, plan, frame_count))
        .unwrap_or_default();
    let frame_requests = frame_prompts
        .into_iter()
        .enumerate()
        .map(|(index, prompt)| MfluxAnimationFrame {
            output_path: workspace
                .join(".sprite-studio/provider-sources")
                .join(format!("mflux-{output_id}-frame-{:03}.png", index + 1)),
            prompt,
            seed: 43 + index as u64,
        })
        .collect::<Vec<_>>();
    let provenance = serde_json::json!({
        "provider": "mflux",
        "model": "z-image-turbo",
        "repository": settings.repository,
        "revision": settings.revision,
        "runtimeVersion": settings.runtime_version,
        "runtime": {
            "python": "3.11",
            "mflux": "0.19.1",
            "mlx": "0.32.0"
        },
        "sourceWidth": source_size,
        "sourceHeight": source_size,
        "targetWidth": generation.width,
        "targetHeight": generation.height,
        "steps": steps,
        "guidance": 0.0,
        "seed": 42,
        "inputMode": generation.image_input_mode,
        "strength": generation.image_strength,
        "animation": requested_animation,
        "frameCount": frame_count,
        "sourceAssetPath": options.source_asset_path,
        "motionPlan": motion_plan,
        "reference": selected_reference.as_ref().map(|reference| serde_json::json!({
            "id": reference.id,
            "name": reference.name,
            "contentHash": reference.content_hash,
        })),
    });
    Ok(Some(MfluxGenerationRequest {
        repository: settings.repository,
        revision: settings.revision,
        output_path,
        prompt: prompt.to_string(),
        width: source_size,
        height: source_size,
        steps,
        guidance: 0.0,
        seed: 42,
        image_path: selected_reference.as_ref().map(|reference| PathBuf::from(&reference.path)),
        image_strength: selected_reference.as_ref().map(|_| generation.image_strength),
        source_master_path,
        frame_requests,
        frame_image_strength: generation.image_strength,
        provenance,
    }))
}

fn write_managed_resources() -> CommandResult<()> {
    let root = cache_root();
    fs::create_dir_all(&root)?;
    let script_path = root.join("install_mflux_runtime.sh");
    fs::write(
        &script_path,
        include_str!("../../scripts/install_mflux_runtime.sh"),
    )?;
    fs::write(
        worker_path(),
        include_str!("../../src-tauri/resources/mflux_worker.py"),
    )?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&script_path, fs::Permissions::from_mode(0o700))?;
        fs::set_permissions(worker_path(), fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn emit_setup(
    app: &AppHandle,
    setup_id: &str,
    event_type: &str,
    stage: &str,
    progress: f64,
    message: impl Into<String>,
) {
    emit_setup_event(
        app,
        MfluxSetupEvent {
            setup_id: setup_id.into(),
            event_type: event_type.into(),
            stage: stage.into(),
            progress,
            message: message.into(),
        },
    );
}

fn emit_setup_event(app: &AppHandle, event: MfluxSetupEvent) {
    let _ = app.emit("mflux-setup-event", event);
}

fn cancelled_setup_event(setup_id: &str) -> MfluxSetupEvent {
    MfluxSetupEvent {
        setup_id: setup_id.into(),
        event_type: "cancelled".into(),
        stage: "cancelled".into(),
        progress: 0.0,
        message: "MFLUX setup cancelled".into(),
    }
}

fn remove_setup(state: &AppState, setup_id: &str) {
    if let Ok(mut setups) = state.mflux_setups.lock() {
        setups.remove(setup_id);
    }
}

fn log_tail(path: &Path) -> String {
    let mut contents = String::new();
    if fs::File::open(path)
        .and_then(|mut file| file.read_to_string(&mut contents))
        .is_err()
    {
        return String::new();
    }
    contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
}

fn cleanup_failed_generation<T>(worker: &mut Option<T>, generated_paths: Vec<PathBuf>) {
    worker.take();
    for path in generated_paths {
        let _ = fs::remove_file(path);
    }
}

struct SetupLogGuard {
    path: PathBuf,
}

impl SetupLogGuard {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for SetupLogGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

async fn run_setup(
    app: AppHandle,
    state: AppState,
    setup_id: String,
    mut cancel_rx: oneshot::Receiver<()>,
) {
    let root = cache_root();
    let script = root.join("install_mflux_runtime.sh");
    let log_path = root.join(format!("setup-{setup_id}.log"));
    let _log_guard = SetupLogGuard::new(log_path.clone());
    emit_setup(
        &app,
        &setup_id,
        "started",
        "preparing",
        0.05,
        "Preparing the managed MFLUX runtime",
    );
    let log = match fs::File::create(&log_path) {
        Ok(log) => log,
        Err(error) => {
            emit_setup(
                &app,
                &setup_id,
                "failed",
                "preparing",
                0.05,
                format!("Could not create the setup log: {error}"),
            );
            remove_setup(&state, &setup_id);
            return;
        }
    };
    let error_log = match log.try_clone() {
        Ok(log) => log,
        Err(error) => {
            emit_setup(
                &app,
                &setup_id,
                "failed",
                "preparing",
                0.05,
                format!("Could not prepare the setup log: {error}"),
            );
            remove_setup(&state, &setup_id);
            return;
        }
    };
    let mut command = Command::new("bash");
    command
        .arg(&script)
        .arg(&root)
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(error_log));
    emit_setup(
        &app,
        &setup_id,
        "progress",
        "installing",
        0.2,
        "Creating the pinned Python 3.11 environment",
    );
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            emit_setup(
                &app,
                &setup_id,
                "failed",
                "installing",
                0.2,
                format!("Could not start the MFLUX installer: {error}"),
            );
            remove_setup(&state, &setup_id);
            return;
        }
    };
    let status = tokio::select! {
        result = child.wait() => result,
        _ = &mut cancel_rx => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            emit_setup_event(&app, cancelled_setup_event(&setup_id));
            remove_setup(&state, &setup_id);
            return;
        }
    };
    match status {
        Ok(status) if status.success() => {
            if let Err(error) = fs::write(runtime_marker_path(), RUNTIME_VERSION) {
                emit_setup(
                    &app,
                    &setup_id,
                    "failed",
                    "verifying",
                    0.9,
                    format!("Runtime installed but could not be verified: {error}"),
                );
            } else {
                emit_setup(
                    &app,
                    &setup_id,
                    "completed",
                    "ready",
                    1.0,
                    "MFLUX runtime is ready. The checkpoint will download on first generation.",
                );
            }
        }
        Ok(status) => {
            let detail = log_tail(&log_path);
            emit_setup(
                &app,
                &setup_id,
                "failed",
                "installing",
                0.2,
                if detail.is_empty() {
                    format!("MFLUX setup failed with {status}")
                } else {
                    format!("MFLUX setup failed with {status}: {detail}")
                },
            );
        }
        Err(error) => emit_setup(
            &app,
            &setup_id,
            "failed",
            "installing",
            0.2,
            format!("MFLUX setup could not finish: {error}"),
        ),
    }
    remove_setup(&state, &setup_id);
}

#[tauri::command]
pub fn get_mflux_settings(state: State<'_, AppState>) -> CommandResult<MfluxSettings> {
    Ok(detect_settings(&state))
}

#[tauri::command]
pub fn save_mflux_settings(
    input: MfluxSettingsInput,
    state: State<'_, AppState>,
) -> CommandResult<MfluxSettings> {
    let input = validate_input(input)?;
    let connection = state
        .db
        .lock()
        .map_err(|_| CommandError::new("database_locked", "Database lock was poisoned"))?;
    connection.execute(
        "INSERT INTO provider_settings(provider,enabled,settings_json,updated_at) VALUES (?1,1,?2,?3) ON CONFLICT(provider) DO UPDATE SET enabled=1,settings_json=excluded.settings_json,updated_at=excluded.updated_at",
        rusqlite::params![
            MFLUX_PROVIDER_KEY,
            serde_json::to_string(&input)
                .map_err(|error| CommandError::new("invalid_mflux_settings", error.to_string()))?,
            Utc::now().to_rfc3339(),
        ],
    )?;
    Ok(status_for(input))
}

#[tauri::command]
pub fn start_mflux_setup(
    app: AppHandle,
    state: State<'_, AppState>,
) -> CommandResult<String> {
    if !supported_host() {
        return Err(CommandError::new(
            "mflux_unsupported_host",
            "MFLUX requires Apple Silicon macOS and cannot be installed on this host.",
        ));
    }
    write_managed_resources()?;
    let setup_id = Uuid::new_v4().to_string();
    let (cancel_tx, cancel_rx) = oneshot::channel();
    state
        .mflux_setups
        .lock()
        .map_err(|_| CommandError::new("process_error", "MFLUX setup registry is unavailable"))?
        .insert(setup_id.clone(), cancel_tx);
    let task_state = state.inner().clone();
    let task_app = app.clone();
    let task_setup_id = setup_id.clone();
    tauri::async_runtime::spawn(async move {
        run_setup(task_app, task_state, task_setup_id, cancel_rx).await;
    });
    Ok(setup_id)
}

#[tauri::command]
pub fn cancel_mflux_setup(
    setup_id: String,
    state: State<'_, AppState>,
) -> CommandResult<()> {
    let sender = state
        .mflux_setups
        .lock()
        .map_err(|_| CommandError::new("process_error", "MFLUX setup registry is unavailable"))?
        .remove(&setup_id);
    if let Some(sender) = sender {
        let _ = sender.send(());
        Ok(())
    } else {
        Err(CommandError::new(
            "request_not_found",
            "The MFLUX setup is no longer running",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        animation_requested, cancelled_setup_event, cleanup_failed_generation, frame_prompts,
        generation_allowed, source_master_path, source_size_and_steps, validate_input,
        validate_reference_selection, validate_repository, validate_revision, SetupLogGuard,
        DEFAULT_REPOSITORY, DEFAULT_REVISION,
    };
    use crate::models::{MotionPhase, MotionPlan, ProviderRequestOptions};
    use std::path::Path;

    #[test]
    fn accepts_the_pinned_default_checkpoint() {
        let settings = validate_input(super::MfluxSettingsInput {
            repository: DEFAULT_REPOSITORY.into(),
            revision: DEFAULT_REVISION.into(),
        })
        .expect("default checkpoint should be valid");
        assert_eq!(settings.repository, DEFAULT_REPOSITORY);
        assert_eq!(settings.revision, DEFAULT_REVISION);
    }

    #[test]
    fn rejects_mutable_or_malformed_revisions() {
        assert!(validate_revision("main").is_err());
        assert!(validate_revision("not-a-commit-hash").is_err());
        assert!(validate_revision(&"a".repeat(40)).is_ok());
    }

    #[test]
    fn rejects_repository_path_traversal() {
        assert!(validate_repository("../private").is_err());
        assert!(validate_repository("owner/model").is_ok());
    }

    #[test]
    fn maps_quality_to_the_pinned_source_size_and_step_budget() {
        assert_eq!(source_size_and_steps("low"), (512, 4));
        assert_eq!(source_size_and_steps("mid"), (1024, 9));
        assert_eq!(source_size_and_steps("high"), (1024, 9));
        assert_eq!(source_size_and_steps("custom"), (1024, 9));
    }

    #[test]
    fn allows_mflux_for_static_and_animation_generation() {
        let mut options = ProviderRequestOptions {
            command: Some("sprite".into()),
            native_rig_master_only: false,
            ..ProviderRequestOptions::default()
        };
        assert!(generation_allowed(&options, "make a sprite"));
        options.command = Some("rig".into());
        assert!(!generation_allowed(&options, "/rig this sprite"));
        options.command = Some("animate".into());
        assert!(generation_allowed(&options, "/animate a walk cycle"));
        options.native_rig_master_only = true;
        assert!(generation_allowed(&options, "create a walking warrior"));
        options.command = None;
        assert!(generation_allowed(&options, "create a walking warrior"));
        options.native_rig_master_only = false;
        assert!(!generation_allowed(&options, "tell me about sprites"));
    }

    #[test]
    fn detects_animation_from_commands_and_motion_language() {
        let mut options = ProviderRequestOptions {
            command: Some("animate".into()),
            ..ProviderRequestOptions::default()
        };
        assert!(animation_requested(&options, "/animate a walk cycle"));
        options.command = None;
        assert!(animation_requested(&options, "create a looping idle animation"));
        assert!(!animation_requested(&options, "create a static treasure chest"));
    }

    #[test]
    fn given_multiple_active_references_when_none_is_focused_then_rejects() {
        let references = vec!["first".into(), "second".into()];
        let error = validate_reference_selection(Some("first"), &references, None)
            .expect_err("multiple active references should require focus");
        assert_eq!(error.code, "mflux_reference_required");
    }

    #[test]
    fn given_one_active_reference_when_it_is_selected_then_accepts() {
        let references = vec!["only".into()];
        assert_eq!(
            validate_reference_selection(Some("only"), &references, None)
                .expect("the only active reference should be accepted"),
            "only"
        );
    }

    #[test]
    fn given_setup_early_exit_when_log_guard_drops_then_removes_log() {
        let path = std::env::temp_dir().join(format!("sprite-studio-mflux-test-{}.log", uuid::Uuid::new_v4()));
        std::fs::write(&path, "setup failure").expect("test log should be created");
        {
            let _guard = SetupLogGuard::new(path.clone());
            assert!(path.is_file());
        }
        assert!(!path.exists());
    }

    #[test]
    fn given_generation_failure_when_cleanup_runs_then_discards_worker_and_outputs() {
        let path = std::env::temp_dir().join(format!("sprite-studio-mflux-test-{}.png", uuid::Uuid::new_v4()));
        std::fs::write(&path, "partial output").expect("partial output should be created");
        let mut worker = Some("broken worker");

        cleanup_failed_generation(&mut worker, vec![path.clone()]);

        assert!(worker.is_none());
        assert!(!path.exists());
    }

    #[test]
    fn given_setup_cancellation_when_event_is_built_then_reports_cancelled() {
        let event = cancelled_setup_event("setup-123");

        assert_eq!(event.setup_id, "setup-123");
        assert_eq!(event.event_type, "cancelled");
        assert_eq!(event.stage, "cancelled");
        assert_eq!(event.progress, 0.0);
    }

    #[test]
    fn builds_ordered_frame_prompts_from_motion_phases() {
        let plan = MotionPlan {
            frame_mode: "fixed".into(),
            selected_frame_count: 3,
            minimum_frame_count: 3,
            maximum_frame_count: 3,
            fps: 12,
            looping: true,
            allow_interpolation: false,
            allow_auto_adjust: false,
            explanation: "repeatable motion".into(),
            phases: vec![
                MotionPhase {
                    name: "anticipation".into(),
                    description: "weight shifts backward".into(),
                    frame_count: 1,
                    timing_weight: 1.0,
                },
                MotionPhase {
                    name: "contact".into(),
                    description: "front foot contacts the ground".into(),
                    frame_count: 2,
                    timing_weight: 1.0,
                },
            ],
        };

        let prompts = frame_prompts("Create a walking character", &plan, 3);

        assert_eq!(prompts.len(), 3);
        assert!(prompts[0].contains("MFLUX ANIMATION FRAME 1/3"));
        assert!(prompts[0].contains("weight shifts backward"));
        assert!(prompts[1].contains("MFLUX ANIMATION FRAME 2/3"));
        assert!(prompts[1].contains("front foot contacts the ground"));
        assert!(prompts[2].contains("MFLUX ANIMATION FRAME 3/3"));
        assert!(prompts[2].contains("front foot contacts the ground"));
    }

    #[test]
    fn rejects_unscoped_or_missing_animation_sources() {
        let workspace = Path::new("/tmp/sprite-maker-test-workspace");

        assert!(source_master_path(workspace, Some("/tmp/master.png")).is_err());
        assert!(source_master_path(workspace, Some("../master.png")).is_err());
        assert!(source_master_path(workspace, Some("references/master.png")).is_err());
        assert!(source_master_path(workspace, Some("assets/master.png")).is_err());
    }
}
