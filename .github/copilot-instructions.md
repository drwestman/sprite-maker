# Sprite Studio Copilot Instructions

Sprite Studio is a local-first desktop application for generating, rigging, testing, and exporting 2D game art. It is a Tauri 2 application with a Svelte 5 frontend and a Rust backend. Read `README.md` for product behavior and workspace/file-format details, and `CONTRIBUTING.md` for contribution and pull-request expectations.

## Build, test, and lint

The repository uses Bun for the frontend and Cargo for the native backend. Install the locked dependency set before running checks:

```bash
bun install --frozen-lockfile
```

Frontend development and validation:

```bash
bun run dev                 # Vite development server on Tauri's port (1420)
bun run check               # SvelteKit sync plus svelte-check/TypeScript validation
bun test                    # All Bun tests
bun test scripts/markdown-links.test.ts
bun test scripts/markdown-links.test.ts -t "routes web URLs outside the desktop webview"
bun run build               # Static SvelteKit build consumed by Tauri
bun tauri dev               # Run the desktop app in development
```

Native validation:

```bash
cargo test --manifest-path src-tauri/Cargo.toml
cargo test --manifest-path src-tauri/Cargo.toml generation_option_tests::omitted_frame_policy_defaults_to_full_auto_range
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
```

`make verify` runs the frontend check, Bun tests, and Rust tests. The pull-request baseline also runs Clippy separately. For a local desktop release, use `make release`; it verifies, builds, and collects host-native artifacts below `release-artifacts/<version>/<platform>`. `make release-macos` builds the universal macOS bundle, and `make publish-macos TAG=v<version>` uploads it to an existing GitHub release.

## Architecture

### Frontend shell and feature components

- `src/routes/+page.svelte` is the single application shell. It owns workspace, worktree, conversation, asset, animation, rig, provider, and tab selection state, coordinates cross-feature refreshes, and listens for provider/job events.
- `src/routes/+layout.ts` disables SSR. SvelteKit uses `adapter-static` with an `index.html` fallback because the app runs inside Tauri rather than behind a Node server.
- `src/lib/components/` contains feature views such as chat, asset browsing, animation, rigging, terrain, VFX, sheets, packs, references, and settings. Components generally receive data and callbacks from the shell, then own their feature-local loading and interaction state.
- `src/lib/api.ts` is the typed frontend IPC boundary. It wraps every Tauri `invoke` call and uses the shared wire types in `src/lib/types.ts`. Pure frontend policy and normalization lives in focused modules such as `generation-profiles.ts`, `message-generations.ts`, `sprite-groups.ts`, and `markdown-links.ts`.

### Rust/Tauri backend

- `src-tauri/src/lib.rs` creates `AppState` with the SQLite connection and request-cancellation map, initializes the app-data database, and registers every `#[tauri::command]` in `tauri::generate_handler!`.
- Backend modules are organized by domain: workspace/projects, worktrees, conversations, references, assets, animations, templates, terrain, jobs, quality, rigging, providers, Ollama, motion planning, settings, packs, and backups.
- `src-tauri/src/models.rs` defines the serialized command/event data shared with TypeScript. `src-tauri/src/database.rs` owns schema creation and migrations. `src-tauri/src/error.rs` defines the serializable `{ code, message }` command error shape.
- SQLite stores app metadata such as projects, chats, assets, versions, animations, jobs, settings, and quality reports. The selected project directory remains the source of user-facing image files and exports. `workspace.rs` initializes the standard project folders and installs bundled tools under `.sprite-studio/`.

### Events and long-running work

- Synchronous UI operations use typed `api.*` calls over Tauri IPC.
- Provider streaming uses the `provider-event` Tauri event. Background rendering, quality analysis, VFX, and sheet generation use the `job-event` event and the `BackgroundJob` model.
- Listeners must filter events by the relevant conversation, project, worktree, and request/job identity, then unsubscribe in the `onMount` cleanup function.
- Provider and generation flow is split across `providers.rs`, `sprite_harness.rs`, and `motion_planner.rs`. Providers may be installed CLIs or configured image/Ollama endpoints; provider capability discovery controls which UI options are offered.
- A completed generation is accepted through the fresh generation manifest and fingerprint, not by guessing paths from assistant prose. The frontend scans/registers those files, links them to the active worktree, creates an animation for ordered multi-frame output, stores message metadata, and queues native quality analysis.
- `rig.rs` renders deterministic animations from points, capsule bones, transforms, contacts, and parent chains. `quality.rs` analyzes animation continuity, alignment, transparency, identity, and loop behavior. `jobs.rs` handles cancellable asynchronous rendering and export work.

## Repository-specific conventions

### Keep the IPC contract synchronized

When adding or changing a command, update all of these together:

1. The Rust command function and its `#[tauri::command]` signature.
2. Module declarations and the command list in `src-tauri/src/lib.rs`.
3. The Rust model/error/event types.
4. The matching `src/lib/types.ts` type and `src/lib/api.ts` wrapper.
5. The consuming Svelte component or shell refresh path.

Rust models use `#[serde(rename_all = "camelCase")]`, so the TypeScript side should use camelCase fields. Preserve this wire shape rather than adding ad hoc conversions in components.

### Backend command and data rules

- Return `CommandResult<T>` and surface failures as `CommandError` with a stable code and user-facing message. The frontend uses `errorMessage()` to decode these errors.
- Validate command inputs at the Rust boundary before touching the database or filesystem. Use `workspace_path()` for project paths and parameterized SQLite queries.
- Keep database access under the `AppState.db` mutex and release the lock before emitting events or performing expensive filesystem/image work.
- Persist UI preferences through the settings commands with namespaced keys such as `activeWorkspaceId`, `active-worktree:<workspace>`, `conversation-style:<conversation>`, and `conversation-generation:<conversation>`.

### Svelte state and asynchronous behavior

- This codebase uses Svelte 5 runes (`$state`, `$derived`, `$effect`) and event attributes such as `onclick`; follow the surrounding component style instead of introducing a different state/event pattern.
- Keep cross-tab and cross-worktree state transitions in `+page.svelte`; keep feature-specific refreshes in the owning component.
- Use the existing selection-generation guards (`workspaceSelection` and `conversationSelection`) when awaiting data during project/chat switches so stale responses cannot overwrite a newer selection.
- On errors, use the existing callback/toast flow and `errorMessage()` rather than swallowing failures or returning success-shaped fallbacks.

### Asset and generation invariants

- Workspace images and exports are ordinary files; database rows register and describe them. Asset revisions are content-hashed and changes are non-destructive.
- Treat the generation manifest as the authoritative provider-to-UI handoff. Do not infer a completed artifact from response text alone, reuse a stale manifest, or silently turn an explicit multi-frame animation into a completed static result.
- Preserve frame order, canvas dimensions, transparency, pivot/ground-line behavior, and animation identity when creating or repairing frames. Queue quality analysis after newly saved animations or rig renders.
- Rig-only animation is deterministic native rendering. AI polish or full-redraw flows must keep the rig-rendered pose, timing, placement, and near/far limb identity authoritative.

### Tests and source placement

- Bun tests live in `scripts/*.test.ts` and exercise pure helpers imported from `src/lib`.
- Rust unit and integration-style tests are kept inline in the relevant `src-tauri/src/*.rs` module under `#[cfg(test)]`.
- Prefer extending an existing domain helper/test module when the behavior belongs there; keep frontend policy out of Svelte markup when it can be tested as a pure TypeScript function.

### Contribution constraints

Keep pull requests focused and preserve unrelated user changes. Do not commit credentials, private workspace data, generated user assets, unlicensed third-party material, binaries, or local release output. Visible UI changes should include screenshots or a short recording, and changes involving migrations, file formats, providers, or platform behavior should call out compatibility and platform-specific risk.
