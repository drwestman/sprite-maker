use super::{
    discovery::provider_process_path, external_source::generate_external_source,
    image_providers::StoredImageProvider,
};
use crate::{
    animations::{export_animation_inner, list_animations_inner, load_animation_by_id},
    assets::{get_asset, list_assets_inner, scan_generation_assets_inner},
    error::{CommandError, CommandResult},
    jobs::{load_job, queue_procedural_vfx_inner, queue_sprite_sheet_inner},
    mflux::{generate_outputs, MfluxGenerationRequest},
    models::{GenerationOptions, ProceduralVfxInput, SpriteSheetInput},
    ollama_logging,
    packs::list_asset_packs_inner,
    quality::{get_quality_report_inner, queue_quality_analysis_inner},
    rig::list_rigs_inner,
    AppState,
};
use chrono::Utc;
use rusqlite::OptionalExtension;
use serde_json::{json, Value};
use std::{
    fs,
    path::{Component, Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};
use tokio::process::Command;
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::ollama_transport::{OllamaToolDefinition, OllamaToolFunctionDefinition};

const MAX_FILE_BYTES: u64 = 256 * 1024;
const MAX_TOOL_OUTPUT_BYTES: usize = 32 * 1024;
const MAX_SEARCH_RESULTS: usize = 200;
const MAX_DIRECTORY_ENTRIES: usize = 200;
const MAX_COMMAND_TIMEOUT_MS: u64 = 120_000;

pub(crate) struct OllamaToolContext<'a> {
    pub state: &'a AppState,
    pub workspace_id: &'a str,
    pub worktree_id: Option<&'a str>,
    pub workspace: &'a Path,
    pub generation: Option<&'a GenerationOptions>,
    pub image_provider: Option<&'a StoredImageProvider>,
    pub mflux_generation: Option<&'a MfluxGenerationRequest>,
    pub image_prompt: &'a str,
    pub command: Option<&'a str>,
}

pub(crate) fn definitions() -> Vec<OllamaToolDefinition> {
    vec![
        function(
            "read_file",
            "Read a UTF-8 text file inside the selected Sprite Studio workspace.",
            json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                    "maxBytes": {"type": "integer", "minimum": 1, "maximum": MAX_FILE_BYTES}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "list_directory",
            "List files and directories inside the selected workspace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative directory path; defaults to the workspace root"},
                    "recursive": {"type": "boolean"},
                    "maxEntries": {"type": "integer", "minimum": 1, "maximum": MAX_DIRECTORY_ENTRIES}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "search_files",
            "Search UTF-8 text files inside the workspace for a literal string.",
            json!({
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "path": {"type": "string"},
                    "maxResults": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_RESULTS}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "write_file",
            "Create or overwrite a UTF-8 text file inside the workspace.",
            json!({
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "createParents": {"type": "boolean"}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "edit_file",
            "Replace an exact text range in a UTF-8 file inside the workspace.",
            json!({
                "type": "object",
                "required": ["path", "oldText", "newText"],
                "properties": {
                    "path": {"type": "string"},
                    "oldText": {"type": "string"},
                    "newText": {"type": "string"},
                    "replaceAll": {"type": "boolean"}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "move_path",
            "Move or rename a file or directory inside the workspace.",
            json!({
                "type": "object",
                "required": ["from", "to"],
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "delete_path",
            "Delete a file or empty directory inside the workspace.",
            json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "run_command",
            "Run a shell command with the selected workspace as its working directory.",
            json!({
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string", "minLength": 1},
                    "cwd": {"type": "string"},
                    "timeoutMs": {"type": "integer", "minimum": 1, "maximum": MAX_COMMAND_TIMEOUT_MS}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "generate_image",
            "Generate a Sprite Studio source image or ordered animation frames with the configured direct image provider, then register the output manifest. Use characters, creatures, terrain, props, or effects for category.",
            json!({
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Optional refinement; the current Sprite Studio request remains authoritative"},
                    "name": {"type": "string", "description": "Stable asset name without an extension"},
                    "category": {"type": "string", "description": "Asset category directory: characters, creatures, terrain, props, or effects"},
                    "fps": {"type": "integer", "minimum": 1, "maximum": 60}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "list_assets",
            "List indexed Sprite Studio assets in the current workspace.",
            json!({"type": "object", "additionalProperties": false}),
        ),
        function(
            "list_artifacts",
            "Scan the latest generation manifest and return its registered artifacts.",
            json!({"type": "object", "additionalProperties": false}),
        ),
        function(
            "list_packs",
            "List generated Sprite Studio asset packs in the current workspace.",
            json!({"type": "object", "additionalProperties": false}),
        ),
        function(
            "list_animations",
            "List animations in the selected workspace or active worktree.",
            json!({"type": "object", "additionalProperties": false}),
        ),
        function(
            "list_rigs",
            "List saved rigs in the selected workspace or active worktree.",
            json!({"type": "object", "additionalProperties": false}),
        ),
        function(
            "export_asset",
            "Export an indexed Sprite Studio asset to a PNG and metadata file.",
            json!({
                "type": "object",
                "required": ["assetId"],
                "properties": {
                    "assetId": {"type": "string"}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "export_animation",
            "Export an indexed Sprite Studio animation to a spritesheet and metadata file.",
            json!({
                "type": "object",
                "required": ["animationId"],
                "properties": {
                    "animationId": {"type": "string"},
                    "destination": {"type": "string", "description": "Workspace-relative export directory"}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "quality_report",
            "Read or queue the latest quality report for an animation.",
            json!({
                "type": "object",
                "required": ["animationId"],
                "properties": {
                    "animationId": {"type": "string"},
                    "analyze": {"type": "boolean"}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "get_job",
            "Read a background Sprite Studio job in the selected workspace.",
            json!({
                "type": "object",
                "required": ["jobId"],
                "properties": {
                    "jobId": {"type": "string"}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "queue_sprite_sheet",
            "Queue a spritesheet export for an animation in the selected workspace.",
            json!({
                "type": "object",
                "required": ["animationId", "frameWidth", "frameHeight"],
                "properties": {
                    "animationId": {"type": "string"},
                    "name": {"type": "string"},
                    "layout": {"type": "string", "enum": ["horizontal", "vertical", "grid"]},
                    "frameWidth": {"type": "integer", "minimum": 1, "maximum": 4096},
                    "frameHeight": {"type": "integer", "minimum": 1, "maximum": 4096},
                    "padding": {"type": "integer", "minimum": 0, "maximum": 4096},
                    "spacing": {"type": "integer", "minimum": 0, "maximum": 4096},
                    "columns": {"type": "integer", "minimum": 1, "maximum": 4096},
                    "scale": {"type": "integer", "minimum": 1, "maximum": 8},
                    "transparent": {"type": "boolean"},
                    "alignment": {"type": "string", "enum": ["top_left", "center", "bottom_center"]},
                    "pivotX": {"type": "number"},
                    "pivotY": {"type": "number"}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "queue_procedural_vfx",
            "Queue a procedural VFX animation in the active VFX worktree.",
            json!({
                "type": "object",
                "required": ["name", "effectType", "width", "height", "frames", "fps", "seed"],
                "properties": {
                    "name": {"type": "string"},
                    "effectType": {"type": "string"},
                    "blendMode": {"type": "string"},
                    "width": {"type": "integer", "minimum": 8, "maximum": 1024},
                    "height": {"type": "integer", "minimum": 8, "maximum": 1024},
                    "frames": {"type": "integer", "minimum": 2, "maximum": 64},
                    "fps": {"type": "integer", "minimum": 1, "maximum": 60},
                    "looping": {"type": "boolean"},
                    "seed": {"type": "integer", "minimum": 0}
                },
                "additionalProperties": false
            }),
        ),
        function(
            "get_generation",
            "Read the status of a background Sprite Studio generation request.",
            json!({
                "type": "object",
                "required": ["requestId"],
                "properties": {
                    "requestId": {"type": "string"}
                },
                "additionalProperties": false
            }),
        ),
    ]
}

fn function(name: &str, description: &str, parameters: Value) -> OllamaToolDefinition {
    OllamaToolDefinition {
        kind: "function".into(),
        function: OllamaToolFunctionDefinition {
            name: name.into(),
            description: description.into(),
            parameters,
        },
    }
}

pub(crate) async fn execute(
    context: &OllamaToolContext<'_>,
    name: &str,
    arguments: &Value,
    cancel_rx: &mut oneshot::Receiver<()>,
) -> CommandResult<Value> {
    let started = Instant::now();
    tracing::debug!(
        target: "ollama",
        event = "tool_started",
        tool = %name,
        arguments = %ollama_logging::redact_json_for_log(arguments, Some(context.workspace), None),
        "Ollama tool started"
    );
    let arguments = arguments.as_object().ok_or_else(|| {
        CommandError::new(
            "ollama_tool_arguments",
            "Tool arguments must be a JSON object",
        )
    })?;
    validate_arguments(name, arguments)?;
    let result = match name {
        "read_file" => read_file(context.workspace, arguments),
        "list_directory" => list_directory(context.workspace, arguments),
        "search_files" => search_files(context.workspace, arguments),
        "write_file" => write_file(context.workspace, arguments),
        "edit_file" => edit_file(context.workspace, arguments),
        "move_path" => move_path(context.workspace, arguments),
        "delete_path" => delete_path(context.workspace, arguments),
        "run_command" => run_command(context.workspace, arguments, cancel_rx).await,
        "generate_image" => generate_image(context, arguments, cancel_rx).await,
        "list_assets" => {
            serde_json::to_value(list_assets_inner(context.workspace_id, context.state)?)
                .map_err(|error| CommandError::new("ollama_tool_result", error.to_string()))
        }
        "list_artifacts" => {
            let assets = scan_generation_assets_inner(context.workspace_id, None, context.state)?;
            serde_json::to_value(assets)
                .map_err(|error| CommandError::new("ollama_tool_result", error.to_string()))
        }

        "list_packs" => {
            serde_json::to_value(list_asset_packs_inner(context.workspace_id, context.state)?)
                .map_err(|error| CommandError::new("ollama_tool_result", error.to_string()))
        }
        "list_animations" => serde_json::to_value(list_animations_inner(
            context.workspace_id,
            context.worktree_id,
            context.state,
        )?)
        .map_err(|error| CommandError::new("ollama_tool_result", error.to_string())),
        "list_rigs" => serde_json::to_value(list_rigs_inner(
            context.workspace_id,
            context.worktree_id,
            context.state,
        )?)
        .map_err(|error| CommandError::new("ollama_tool_result", error.to_string())),
        "export_asset" => {
            let asset_id = required_string(arguments, "assetId")?;
            let asset = get_asset(context.state, asset_id)?;
            if asset.workspace_id != context.workspace_id {
                return Err(CommandError::new(
                    "ollama_tool_scope",
                    "The requested asset is outside the selected workspace",
                ));
            }
            let result = crate::assets::export_asset_inner(asset_id, context.state)?;
            serde_json::to_value(result)
                .map_err(|error| CommandError::new("ollama_tool_result", error.to_string()))
        }
        "export_animation" => {
            let animation_id = required_string(arguments, "animationId")?;
            let animation = load_animation_by_id(context.state, animation_id)?;
            if animation.workspace_id != context.workspace_id {
                return Err(CommandError::new(
                    "ollama_tool_scope",
                    "The requested animation is outside the selected workspace",
                ));
            }
            let destination = optional_string(arguments, "destination");
            let result = export_animation_inner(
                animation_id,
                destination.map(str::to_owned),
                context.state,
            )?;
            serde_json::to_value(result)
                .map_err(|error| CommandError::new("ollama_tool_result", error.to_string()))
        }
        "quality_report" => {
            let animation_id = required_string(arguments, "animationId")?;
            let animation_workspace = {
                let connection = context.state.db.lock().map_err(|_| {
                    CommandError::new("database_locked", "Database lock was poisoned")
                })?;
                connection
                    .query_row(
                        "SELECT workspace_id FROM animations WHERE id=?1",
                        [animation_id],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()?
            };
            if animation_workspace.as_deref() != Some(context.workspace_id) {
                return Err(CommandError::new(
                    "ollama_tool_scope",
                    "The requested animation is outside the selected workspace",
                ));
            }
            let existing = get_quality_report_inner(animation_id, context.state)?;
            if optional_bool(arguments, "analyze").unwrap_or(false)
                || existing
                    .as_ref()
                    .is_none_or(|report| report.status != "completed")
            {
                let job =
                    queue_quality_analysis_inner(animation_id.to_string(), None, context.state)?;
                Ok(json!({"jobId": job.id, "status": job.status}))
            } else {
                serde_json::to_value(existing)
                    .map_err(|error| CommandError::new("ollama_tool_result", error.to_string()))
            }
        }
        "get_job" => {
            let job_id = required_string(arguments, "jobId")?;
            let job = load_job(context.state, job_id)?;
            if job.project_id != context.workspace_id {
                return Err(CommandError::new(
                    "ollama_tool_scope",
                    "The requested job is outside the selected workspace",
                ));
            }
            serde_json::to_value(job)
                .map_err(|error| CommandError::new("ollama_tool_result", error.to_string()))
        }
        "queue_sprite_sheet" => {
            let animation_id = required_string(arguments, "animationId")?;
            let animation = load_animation_by_id(context.state, animation_id)?;
            if animation.workspace_id != context.workspace_id {
                return Err(CommandError::new(
                    "ollama_tool_scope",
                    "The requested animation is outside the selected workspace",
                ));
            }
            let input = SpriteSheetInput {
                project_id: context.workspace_id.to_string(),
                worktree_id: context.worktree_id.map(str::to_owned),
                animation_id: animation_id.to_string(),
                name: optional_string(arguments, "name")
                    .unwrap_or("sprite-sheet")
                    .to_string(),
                layout: optional_string(arguments, "layout")
                    .unwrap_or("horizontal")
                    .to_string(),
                frame_width: required_u64(arguments, "frameWidth")? as u32,
                frame_height: required_u64(arguments, "frameHeight")? as u32,
                padding: optional_u64(arguments, "padding").unwrap_or(0) as u32,
                spacing: optional_u64(arguments, "spacing").unwrap_or(0) as u32,
                columns: optional_u64(arguments, "columns").unwrap_or(1) as u32,
                scale: optional_u64(arguments, "scale").unwrap_or(1) as u32,
                transparent: optional_bool(arguments, "transparent").unwrap_or(true),
                alignment: optional_string(arguments, "alignment")
                    .unwrap_or("center")
                    .to_string(),
                pivot_x: optional_f64(arguments, "pivotX").unwrap_or(0.5),
                pivot_y: optional_f64(arguments, "pivotY").unwrap_or(1.0),
            };
            let job = queue_sprite_sheet_inner(input, None, context.state)?;
            serde_json::to_value(job)
                .map_err(|error| CommandError::new("ollama_tool_result", error.to_string()))
        }
        "queue_procedural_vfx" => {
            let worktree_id = context.worktree_id.ok_or_else(|| {
                CommandError::new(
                    "ollama_tool_scope",
                    "Procedural VFX requires an active VFX worktree",
                )
            })?;
            let input = ProceduralVfxInput {
                project_id: context.workspace_id.to_string(),
                worktree_id: worktree_id.to_string(),
                name: required_string(arguments, "name")?.to_string(),
                effect_type: required_string(arguments, "effectType")?.to_string(),
                blend_mode: optional_string(arguments, "blendMode")
                    .unwrap_or("normal")
                    .to_string(),
                width: required_u64(arguments, "width")? as u32,
                height: required_u64(arguments, "height")? as u32,
                frames: required_u64(arguments, "frames")? as u32,
                fps: required_u64(arguments, "fps")? as u32,
                looping: optional_bool(arguments, "looping").unwrap_or(true),
                seed: required_u64(arguments, "seed")?,
            };
            let job = queue_procedural_vfx_inner(input, None, context.state)?;
            serde_json::to_value(job)
                .map_err(|error| CommandError::new("ollama_tool_result", error.to_string()))
        }
        "get_generation" => {
            let request_id = required_string(arguments, "requestId")?;
            match context.state.generation_snapshot(request_id) {
                Some(snapshot) if snapshot.workspace_id == context.workspace_id => {
                    serde_json::to_value(snapshot)
                        .map_err(|error| CommandError::new("ollama_tool_result", error.to_string()))
                }
                Some(_) => Err(CommandError::new(
                    "ollama_tool_scope",
                    "The requested generation is outside the selected workspace",
                )),
                None => Ok(json!({"status": "unknown", "requestId": request_id})),
            }
        }
        _ => Err(CommandError::new(
            "ollama_unknown_tool",
            format!("Unknown Ollama tool: {name}"),
        )),
    };
    match &result {
        Ok(value) => tracing::debug!(
            target: "ollama",
            event = "tool_completed",
            tool = %name,
            duration_ms = started.elapsed().as_millis() as u64,
            result = %ollama_logging::redact_json_for_log(
                value,
                Some(context.workspace),
                None,
            ),
            "Ollama tool completed"
        ),
        Err(error) => tracing::debug!(
            target: "ollama",
            event = "tool_failed",
            tool = %name,
            duration_ms = started.elapsed().as_millis() as u64,
            error_code = %error.code,
            error = %ollama_logging::redact_text(
                &error.message,
                Some(context.workspace),
                None,
            ),
            "Ollama tool failed"
        ),
    }
    result
}

fn validate_arguments(name: &str, arguments: &serde_json::Map<String, Value>) -> CommandResult<()> {
    let allowed = match name {
        "read_file" => &["path", "maxBytes"][..],
        "list_directory" => &["path", "recursive", "maxEntries"][..],
        "search_files" => &["query", "path", "maxResults"][..],
        "write_file" => &["path", "content", "createParents"][..],
        "edit_file" => &["path", "oldText", "newText", "replaceAll"][..],
        "move_path" => &["from", "to"][..],
        "delete_path" => &["path", "recursive"][..],
        "run_command" => &["command", "cwd", "timeoutMs"][..],
        "generate_image" => &["prompt", "name", "category", "fps"][..],
        "list_assets" | "list_artifacts" | "list_packs" | "list_animations" | "list_rigs" => {
            &[][..]
        }
        "export_asset" => &["assetId"][..],
        "export_animation" => &["animationId", "destination"][..],
        "quality_report" => &["animationId", "analyze"][..],
        "get_job" => &["jobId"][..],
        "queue_sprite_sheet" => &[
            "animationId",
            "name",
            "layout",
            "frameWidth",
            "frameHeight",
            "padding",
            "spacing",
            "columns",
            "scale",
            "transparent",
            "alignment",
            "pivotX",
            "pivotY",
        ][..],
        "queue_procedural_vfx" => &[
            "name",
            "effectType",
            "blendMode",
            "width",
            "height",
            "frames",
            "fps",
            "looping",
            "seed",
        ][..],
        "get_generation" => &["requestId"][..],
        _ => {
            return Err(CommandError::new(
                "ollama_unknown_tool",
                format!("Unknown Ollama tool: {name}"),
            ))
        }
    };
    if let Some(key) = arguments
        .keys()
        .find(|key| !allowed.contains(&key.as_str()))
    {
        return Err(CommandError::new(
            "ollama_tool_arguments",
            format!("{name} does not accept the `{key}` argument"),
        ));
    }
    for key in [
        "path",
        "query",
        "oldText",
        "from",
        "to",
        "command",
        "cwd",
        "prompt",
        "name",
        "category",
        "assetId",
        "animationId",
        "requestId",
        "destination",
    ] {
        if arguments.contains_key(key) && arguments[key].as_str().is_none() {
            return Err(CommandError::new(
                "ollama_tool_arguments",
                format!("{name}.{key} must be a string"),
            ));
        }
    }
    for key in [
        "recursive",
        "createParents",
        "replaceAll",
        "analyze",
        "transparent",
        "looping",
    ] {
        if arguments.contains_key(key) && arguments[key].as_bool().is_none() {
            return Err(CommandError::new(
                "ollama_tool_arguments",
                format!("{name}.{key} must be a boolean"),
            ));
        }
    }
    for key in [
        "maxBytes",
        "maxEntries",
        "maxResults",
        "timeoutMs",
        "fps",
        "frameWidth",
        "frameHeight",
        "padding",
        "spacing",
        "columns",
        "scale",
        "width",
        "height",
        "frames",
        "seed",
    ] {
        if arguments.contains_key(key) && arguments[key].as_u64().is_none() {
            return Err(CommandError::new(
                "ollama_tool_arguments",
                format!("{name}.{key} must be a positive integer"),
            ));
        }
    }
    if let Some(value) = arguments.get("maxBytes").and_then(Value::as_u64) {
        if !(1..=MAX_FILE_BYTES).contains(&value) {
            return Err(CommandError::new(
                "ollama_tool_arguments",
                "maxBytes is outside the allowed range",
            ));
        }
    }
    if let Some(value) = arguments.get("maxEntries").and_then(Value::as_u64) {
        if !(1..=MAX_DIRECTORY_ENTRIES as u64).contains(&value) {
            return Err(CommandError::new(
                "ollama_tool_arguments",
                "maxEntries is outside the allowed range",
            ));
        }
    }
    if let Some(value) = arguments.get("maxResults").and_then(Value::as_u64) {
        if !(1..=MAX_SEARCH_RESULTS as u64).contains(&value) {
            return Err(CommandError::new(
                "ollama_tool_arguments",
                "maxResults is outside the allowed range",
            ));
        }
    }
    if let Some(value) = arguments.get("timeoutMs").and_then(Value::as_u64) {
        if !(1..=MAX_COMMAND_TIMEOUT_MS).contains(&value) {
            return Err(CommandError::new(
                "ollama_tool_arguments",
                "timeoutMs is outside the allowed range",
            ));
        }
    }
    if let Some(value) = arguments.get("fps").and_then(Value::as_u64) {
        if !(1..=60).contains(&value) {
            return Err(CommandError::new(
                "ollama_tool_arguments",
                "fps is outside the allowed range",
            ));
        }
    }
    for (key, range) in [
        ("frameWidth", 1..=4096),
        ("frameHeight", 1..=4096),
        ("padding", 0..=4096),
        ("spacing", 0..=4096),
        ("columns", 1..=4096),
        ("scale", 1..=8),
        ("width", 8..=1024),
        ("height", 8..=1024),
        ("frames", 2..=64),
    ] {
        if let Some(value) = arguments.get(key).and_then(Value::as_u64) {
            if !range.contains(&value) {
                return Err(CommandError::new(
                    "ollama_tool_arguments",
                    format!("{key} is outside the allowed range"),
                ));
            }
        }
    }
    for key in ["pivotX", "pivotY"] {
        if arguments.contains_key(key) && arguments[key].as_f64().is_none() {
            return Err(CommandError::new(
                "ollama_tool_arguments",
                format!("{key} must be a number"),
            ));
        }
    }
    for key in [
        "path",
        "query",
        "oldText",
        "from",
        "to",
        "command",
        "prompt",
        "name",
        "category",
        "assetId",
        "animationId",
        "requestId",
        "destination",
    ] {
        if arguments
            .get(key)
            .is_some_and(|value| value.as_str().is_some_and(|text| text.trim().is_empty()))
        {
            return Err(CommandError::new(
                "ollama_tool_arguments",
                format!("{name}.{key} must not be empty"),
            ));
        }
    }
    Ok(())
}

async fn generate_image(
    context: &OllamaToolContext<'_>,
    arguments: &serde_json::Map<String, Value>,
    cancel_rx: &mut oneshot::Receiver<()>,
) -> CommandResult<Value> {
    let Some(generation) = context.generation else {
        return Err(CommandError::new(
            "ollama_generation_profile_required",
            "Image generation requires a Sprite Studio generation profile",
        ));
    };
    let output = if let Some(request) = context.mflux_generation {
        generate_outputs(context.state, request, cancel_rx).await?
    } else if let Some(provider) = context.image_provider {
        let prompt = optional_string(arguments, "prompt")
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(context.image_prompt);
        let future = generate_external_source(context.workspace, provider, prompt);
        tokio::select! {
            _ = &mut *cancel_rx => return Err(CommandError::new("request_cancelled", "Request cancelled")),
            result = future => {
                let master_path = result?;
                crate::mflux::MfluxGeneratedOutput {
                    master_path,
                    frame_paths: Vec::new(),
                }
            }
        }
    } else {
        return Err(CommandError::new(
            "ollama_image_provider_required",
            "Direct Ollama image generation requires MFLUX or a configured OpenAI-compatible image provider",
        ));
    };
    register_generated_output(context, arguments, generation, output)
}

fn register_generated_output(
    context: &OllamaToolContext<'_>,
    arguments: &serde_json::Map<String, Value>,
    generation: &GenerationOptions,
    output: crate::mflux::MfluxGeneratedOutput,
) -> CommandResult<Value> {
    let name = optional_string(arguments, "name")
        .map(slug)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "ollama-output".into());
    let category = optional_string(arguments, "category")
        .map(slug)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            match context.command {
                Some("character") => "characters",
                Some("effect") => "effects",
                Some("rig") | Some("animate") => "characters",
                _ => "props",
            }
            .into()
        });
    let directory = context.workspace.join("assets").join(&category);
    fs::create_dir_all(&directory)?;
    let output_id = Uuid::new_v4().simple().to_string();
    let stem = format!("{name}-{}", &output_id[..8]);
    let copy_output = |source: &Path, suffix: &str| -> CommandResult<String> {
        if !source.is_file() {
            return Err(CommandError::new(
                "ollama_generation_output",
                format!("Generated image is missing: {}", source.display()),
            ));
        }
        let filename = format!("{stem}{suffix}.png");
        let destination = directory.join(&filename);
        fs::copy(source, &destination)?;
        Ok(format!("assets/{category}/{filename}"))
    };
    let master = copy_output(&output.master_path, "")?;
    let mut frame_files = Vec::new();
    for (index, frame) in output.frame_paths.iter().enumerate() {
        frame_files.push(copy_output(frame, &format!("-{:03}", index + 1))?);
    }
    let files = if frame_files.is_empty() {
        vec![master.clone()]
    } else {
        frame_files
    };
    let manifest = json!({
        "kind": "sprite",
        "name": name,
        "category": category,
        "fps": optional_u64(arguments, "fps").unwrap_or(generation.fps as u64).clamp(1, 60),
        "files": files,
        "generatedAt": Utc::now().to_rfc3339(),
        "source": master,
    });
    let manifest_path = context
        .workspace
        .join(".sprite-studio/last-generation.json");
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest)
            .map_err(|error| CommandError::new("ollama_generation_output", error.to_string()))?,
    )?;
    Ok(json!({
        "manifest": manifest,
        "masterPath": manifest["source"],
        "frameCount": output.frame_paths.len(),
    }))
}

fn slug(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .chars()
        .take(64)
        .collect()
}

fn read_file(root: &Path, arguments: &serde_json::Map<String, Value>) -> CommandResult<Value> {
    let path = resolve_path(root, required_string(arguments, "path")?, false)?;
    let metadata = fs::metadata(&path)?;
    if !metadata.is_file() {
        return Err(CommandError::new(
            "ollama_tool_path",
            "The requested path is not a file",
        ));
    }
    let max_bytes = optional_u64(arguments, "maxBytes")
        .unwrap_or(MAX_FILE_BYTES)
        .min(MAX_FILE_BYTES);
    if metadata.len() > max_bytes {
        return Err(CommandError::new(
            "ollama_tool_output",
            format!("File is larger than the {max_bytes}-byte read limit"),
        ));
    }
    let content = fs::read_to_string(&path).map_err(|error| {
        CommandError::new(
            "ollama_tool_read",
            format!(
                "Could not read {} as UTF-8 text: {error}",
                display_path(root, &path)
            ),
        )
    })?;
    Ok(json!({"path": display_path(root, &path), "content": content}))
}

fn list_directory(root: &Path, arguments: &serde_json::Map<String, Value>) -> CommandResult<Value> {
    let relative = optional_string(arguments, "path").unwrap_or("");
    let directory = resolve_path(root, relative, false)?;
    if !directory.is_dir() {
        return Err(CommandError::new(
            "ollama_tool_path",
            "The requested path is not a directory",
        ));
    }
    let recursive = optional_bool(arguments, "recursive").unwrap_or(false);
    let max_entries = optional_u64(arguments, "maxEntries")
        .unwrap_or(MAX_DIRECTORY_ENTRIES as u64)
        .clamp(1, MAX_DIRECTORY_ENTRIES as u64) as usize;
    let mut entries = Vec::new();
    collect_directory(root, &directory, recursive, max_entries, &mut entries)?;
    Ok(json!({"path": display_path(root, &directory), "entries": entries}))
}

fn collect_directory(
    root: &Path,
    directory: &Path,
    recursive: bool,
    max_entries: usize,
    entries: &mut Vec<Value>,
) -> CommandResult<()> {
    let mut children = fs::read_dir(directory)?
        .filter_map(Result::ok)
        .collect::<Vec<_>>();
    children.sort_by_key(|entry| entry.file_name());
    for entry in children {
        if entries.len() >= max_entries {
            break;
        }
        let path = entry.path();
        if !is_safe_existing_path(root, &path) {
            continue;
        }
        let metadata = entry.metadata()?;
        entries.push(json!({
            "path": display_path(root, &path),
            "kind": if metadata.is_dir() { "directory" } else { "file" },
            "bytes": metadata.len(),
        }));
        if recursive && metadata.is_dir() {
            collect_directory(root, &path, recursive, max_entries, entries)?;
        }
    }
    Ok(())
}

fn search_files(root: &Path, arguments: &serde_json::Map<String, Value>) -> CommandResult<Value> {
    let query = required_string(arguments, "query")?;
    let start = resolve_path(
        root,
        optional_string(arguments, "path").unwrap_or(""),
        false,
    )?;
    let max_results = optional_u64(arguments, "maxResults")
        .unwrap_or(MAX_SEARCH_RESULTS as u64)
        .clamp(1, MAX_SEARCH_RESULTS as u64) as usize;
    let mut results = Vec::new();
    search_directory(root, &start, query, max_results, &mut results)?;
    Ok(json!({"query": query, "matches": results}))
}

fn search_directory(
    root: &Path,
    directory: &Path,
    query: &str,
    max_results: usize,
    results: &mut Vec<Value>,
) -> CommandResult<()> {
    if results.len() >= max_results {
        return Ok(());
    }
    let mut children = fs::read_dir(directory)?
        .filter_map(Result::ok)
        .collect::<Vec<_>>();
    children.sort_by_key(|entry| entry.file_name());
    for entry in children {
        if results.len() >= max_results {
            break;
        }
        let path = entry.path();
        if !is_safe_existing_path(root, &path) {
            continue;
        }
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            search_directory(root, &path, query, max_results, results)?;
            continue;
        }
        if metadata.len() > MAX_FILE_BYTES {
            continue;
        }
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        for (line_number, line) in content.lines().enumerate() {
            if line.contains(query) {
                results.push(json!({
                    "path": display_path(root, &path),
                    "line": line_number + 1,
                    "text": truncate(line, MAX_TOOL_OUTPUT_BYTES / 8),
                }));
                if results.len() >= max_results {
                    break;
                }
            }
        }
    }
    Ok(())
}

fn write_file(root: &Path, arguments: &serde_json::Map<String, Value>) -> CommandResult<Value> {
    let path = resolve_path(root, required_string(arguments, "path")?, true)?;
    let content = required_text(arguments, "content")?;
    if content.len() > MAX_TOOL_OUTPUT_BYTES * 8 {
        return Err(CommandError::new(
            "ollama_tool_output",
            "Refusing to write a file larger than the tool limit",
        ));
    }
    if optional_bool(arguments, "createParents").unwrap_or(false) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    } else if path.parent().is_some_and(|parent| !parent.is_dir()) {
        return Err(CommandError::new(
            "ollama_tool_path",
            "Parent directory does not exist; set createParents to true",
        ));
    }
    fs::write(&path, content)?;
    Ok(json!({"path": display_path(root, &path), "bytes": content.len()}))
}

fn edit_file(root: &Path, arguments: &serde_json::Map<String, Value>) -> CommandResult<Value> {
    let path = resolve_path(root, required_string(arguments, "path")?, false)?;
    let mut content = fs::read_to_string(&path)?;
    let old_text = required_string(arguments, "oldText")?;
    let new_text = required_text(arguments, "newText")?;
    let replace_all = optional_bool(arguments, "replaceAll").unwrap_or(false);
    let occurrences = content.matches(old_text).count();
    if occurrences == 0 {
        return Err(CommandError::new(
            "ollama_tool_edit",
            "The exact oldText was not found in the file",
        ));
    }
    if !replace_all && occurrences != 1 {
        return Err(CommandError::new(
            "ollama_tool_edit",
            "oldText matched multiple locations; provide a more specific range or set replaceAll",
        ));
    }
    if replace_all {
        content = content.replace(old_text, new_text);
    } else if let Some(index) = content.find(old_text) {
        content.replace_range(index..index + old_text.len(), new_text);
    }
    fs::write(&path, content)?;
    Ok(
        json!({"path": display_path(root, &path), "replacements": if replace_all { occurrences } else { 1 }}),
    )
}

fn move_path(root: &Path, arguments: &serde_json::Map<String, Value>) -> CommandResult<Value> {
    let from = resolve_path(root, required_string(arguments, "from")?, false)?;
    let to = resolve_path(root, required_string(arguments, "to")?, true)?;
    reject_workspace_root(root, &from)?;
    if to.exists() {
        return Err(CommandError::new(
            "ollama_tool_path",
            "The destination already exists",
        ));
    }
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::rename(&from, &to)?;
    Ok(json!({
        "from": display_path(root, &from),
        "to": display_path(root, &to),
    }))
}

fn delete_path(root: &Path, arguments: &serde_json::Map<String, Value>) -> CommandResult<Value> {
    let path = resolve_path(root, required_string(arguments, "path")?, false)?;
    reject_workspace_root(root, &path)?;
    let recursive = optional_bool(arguments, "recursive").unwrap_or(false);
    let metadata = fs::metadata(&path)?;
    if metadata.is_dir() {
        if recursive {
            fs::remove_dir_all(&path)?;
        } else {
            fs::remove_dir(&path)?;
        }
    } else {
        fs::remove_file(&path)?;
    }
    Ok(json!({"path": display_path(root, &path), "deleted": true}))
}

async fn run_command(
    root: &Path,
    arguments: &serde_json::Map<String, Value>,
    cancel_rx: &mut oneshot::Receiver<()>,
) -> CommandResult<Value> {
    let command = required_string(arguments, "command")?;
    let cwd = resolve_path(root, optional_string(arguments, "cwd").unwrap_or(""), false)?;
    if !cwd.is_dir() {
        return Err(CommandError::new(
            "ollama_tool_path",
            "Command cwd is not a directory",
        ));
    }
    let timeout_ms = optional_u64(arguments, "timeoutMs")
        .unwrap_or(MAX_COMMAND_TIMEOUT_MS)
        .clamp(1, MAX_COMMAND_TIMEOUT_MS);
    let mut process = if cfg!(windows) {
        let mut process = Command::new("cmd");
        process.args(["/C", command]);
        process
    } else {
        let mut process = Command::new("sh");
        process.args(["-lc", command]);
        process
    };
    process
        .current_dir(&cwd)
        .env("PATH", provider_process_path("ollama"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let output = tokio::select! {
        _ = &mut *cancel_rx => {
            return Err(CommandError::new("request_cancelled", "Request cancelled"));
        }
        result = tokio::time::timeout(Duration::from_millis(timeout_ms), process.output()) => {
            match result {
                Ok(Ok(output)) => output,
                Ok(Err(error)) => return Err(CommandError::new("ollama_tool_command", error.to_string())),
                Err(_) => return Err(CommandError::new("ollama_tool_timeout", "Command exceeded the allowed timeout")),
            }
        }
    };
    Ok(json!({
        "cwd": display_path(root, &cwd),
        "status": output.status.code(),
        "success": output.status.success(),
        "stdout": truncate(&String::from_utf8_lossy(&output.stdout), MAX_TOOL_OUTPUT_BYTES),
        "stderr": truncate(&String::from_utf8_lossy(&output.stderr), MAX_TOOL_OUTPUT_BYTES),
    }))
}

fn resolve_path(root: &Path, value: &str, allow_missing: bool) -> CommandResult<PathBuf> {
    let value = value.trim();
    let relative = Path::new(value);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)))
    {
        return Err(CommandError::new(
            "path_outside_workspace",
            "Tool paths must stay inside the selected workspace",
        ));
    }
    let canonical_root = root.canonicalize().map_err(CommandError::from)?;
    let candidate = canonical_root.join(relative);
    let mut component_path = canonical_root.clone();
    for component in relative.components() {
        if let Component::Normal(part) = component {
            component_path.push(part);
            if fs::symlink_metadata(&component_path)
                .map(|metadata| metadata.file_type().is_symlink())
                .unwrap_or(false)
            {
                return Err(CommandError::new(
                    "path_outside_workspace",
                    "Tool paths may not traverse symbolic links",
                ));
            }
        }
    }
    if candidate.exists() {
        let canonical = candidate.canonicalize().map_err(CommandError::from)?;
        if canonical.starts_with(&canonical_root) {
            return Ok(canonical);
        }
        return Err(CommandError::new(
            "path_outside_workspace",
            "The requested path resolves outside the selected workspace",
        ));
    }
    if !allow_missing {
        return Err(CommandError::new(
            "ollama_tool_path",
            "The requested path does not exist",
        ));
    }
    let mut ancestor = candidate.as_path();
    while !ancestor.exists() {
        ancestor = ancestor.parent().ok_or_else(|| {
            CommandError::new("path_outside_workspace", "The requested path is invalid")
        })?;
    }
    let canonical_ancestor = ancestor.canonicalize().map_err(CommandError::from)?;
    if !canonical_ancestor.starts_with(&canonical_root) {
        return Err(CommandError::new(
            "path_outside_workspace",
            "The requested path resolves outside the selected workspace",
        ));
    }
    Ok(candidate)
}

fn reject_workspace_root(root: &Path, path: &Path) -> CommandResult<()> {
    let canonical_root = root.canonicalize().map_err(CommandError::from)?;
    if path == canonical_root {
        return Err(CommandError::new(
            "ollama_tool_path",
            "The workspace root itself cannot be moved or deleted",
        ));
    }
    Ok(())
}

fn is_safe_existing_path(root: &Path, path: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(root) else {
        return false;
    };
    resolve_path(root, &relative.to_string_lossy(), false).is_ok()
}

fn display_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn required_string<'a>(
    arguments: &'a serde_json::Map<String, Value>,
    key: &str,
) -> CommandResult<&'a str> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| CommandError::new("ollama_tool_arguments", format!("{key} is required")))
}

fn required_text<'a>(
    arguments: &'a serde_json::Map<String, Value>,
    key: &str,
) -> CommandResult<&'a str> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| CommandError::new("ollama_tool_arguments", format!("{key} is required")))
}

fn optional_string<'a>(
    arguments: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Option<&'a str> {
    arguments.get(key).and_then(Value::as_str)
}

fn optional_bool(arguments: &serde_json::Map<String, Value>, key: &str) -> Option<bool> {
    arguments.get(key).and_then(Value::as_bool)
}

fn optional_u64(arguments: &serde_json::Map<String, Value>, key: &str) -> Option<u64> {
    arguments.get(key).and_then(Value::as_u64)
}

fn required_u64(arguments: &serde_json::Map<String, Value>, key: &str) -> CommandResult<u64> {
    arguments
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| CommandError::new("ollama_tool_arguments", format!("{key} is required")))
}

fn optional_f64(arguments: &serde_json::Map<String, Value>, key: &str) -> Option<f64> {
    arguments.get(key).and_then(Value::as_f64)
}

fn truncate(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &value[..end])
}

#[cfg(test)]
mod tests {
    use super::{
        definitions, register_generated_output, reject_workspace_root, resolve_path,
        validate_arguments, OllamaToolContext,
    };
    use crate::{mflux::MfluxGeneratedOutput, models::GenerationOptions, AppState};
    use serde_json::json;
    use std::fs;
    use uuid::Uuid;

    #[test]
    fn definitions_include_workspace_and_studio_tools() {
        let names = definitions()
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"read_file".into()));
        assert!(names.contains(&"run_command".into()));
        assert!(names.contains(&"generate_image".into()));
        assert!(names.contains(&"quality_report".into()));
        assert!(names.contains(&"list_animations".into()));
        assert!(names.contains(&"queue_sprite_sheet".into()));
    }

    #[test]
    fn rejects_unknown_and_malformed_domain_tool_arguments() {
        let unknown =
            serde_json::from_value(json!({"animationId": "animation", "unexpected": true}))
                .unwrap();
        assert_eq!(
            validate_arguments("export_animation", &unknown)
                .expect_err("unknown fields must be rejected")
                .code,
            "ollama_tool_arguments"
        );
        let malformed = serde_json::from_value(json!({
            "animationId": "animation",
            "frameWidth": "64",
            "frameHeight": 64
        }))
        .unwrap();
        assert_eq!(
            validate_arguments("queue_sprite_sheet", &malformed)
                .expect_err("numeric fields must be validated")
                .code,
            "ollama_tool_arguments"
        );
    }

    #[test]
    fn multi_frame_manifest_keeps_the_master_out_of_animation_files() {
        let root = std::env::temp_dir().join(format!("ollama-manifest-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).expect("root");
        let master = root.join("master.png");
        let frame = root.join("frame.png");
        fs::write(&master, b"master").expect("master");
        fs::write(&frame, b"frame").expect("frame");
        let state = AppState::from_connection(
            rusqlite::Connection::open_in_memory().expect("in-memory database"),
        );
        let generation = GenerationOptions {
            quality: "mid".into(),
            width: 64,
            height: 64,
            frames: 2,
            fps: 8,
            frame_mode: "auto".into(),
            min_frames: 2,
            max_frames: 2,
            allow_interpolation: false,
            allow_auto_adjust: true,
            image_input_mode: "text-to-image".into(),
            image_strength: 0.4,
        };
        let context = OllamaToolContext {
            state: &state,
            workspace_id: "workspace",
            worktree_id: None,
            workspace: &root,
            generation: Some(&generation),
            image_provider: None,
            mflux_generation: None,
            image_prompt: "test",
            command: Some("animate"),
        };
        let output = register_generated_output(
            &context,
            &serde_json::Map::new(),
            &generation,
            MfluxGeneratedOutput {
                master_path: master,
                frame_paths: vec![frame],
            },
        )
        .expect("manifest");
        let manifest_path = root.join(".sprite-studio/last-generation.json");
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(manifest_path).expect("manifest file"))
                .expect("manifest JSON");
        assert_eq!(manifest["files"].as_array().map(Vec::len), Some(1));
        assert_eq!(manifest["source"], output["masterPath"]);
        assert_ne!(manifest["files"][0], manifest["source"]);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn workspace_paths_reject_traversal_and_symlink_escape() {
        let root = std::env::temp_dir().join(format!("ollama-tools-{}", Uuid::new_v4()));
        let outside = std::env::temp_dir().join(format!("ollama-tools-outside-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).expect("root");
        fs::create_dir_all(&outside).expect("outside");
        assert!(resolve_path(&root, "../outside", false).is_err());
        assert!(resolve_path(&root, "/etc/passwd", false).is_err());
        assert!(
            reject_workspace_root(&root, &root.canonicalize().expect("canonical root")).is_err()
        );
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside, root.join("link")).expect("symlink");
            assert!(resolve_path(&root, "link/file.txt", true).is_err());
        }
        fs::remove_dir_all(root).ok();
        fs::remove_dir_all(outside).ok();
    }
}
