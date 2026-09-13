<script lang="ts">
  import { onMount } from "svelte";
  import { listen } from "@tauri-apps/api/event";
  import { X, Settings, Bot, Image, Palette, CheckCircle2, CircleSlash2, RefreshCw, Plus, Trash2, PlugZap, ShieldCheck, Terminal, ArrowRight, Download, LogIn } from "lucide-svelte";
  import { api } from "$lib/api";
  import type { ImageProviderInput, MfluxSettings, MfluxSettingsInput, MfluxSetupEvent, OllamaModel, OllamaSettings, OllamaSettingsInput, ProviderConnectionTest, ProviderStatus } from "$lib/types";
  import StylePicker from "$lib/components/StylePicker.svelte";
  import type { StylePresetId } from "$lib/style-presets";
  import type { CustomArtStyle } from "$lib/library-types";

  let { providers, defaultProvider, generationProvider, theme, workspaceStyle, customStyles = [], onDefaultProvider, onGenerationProvider, onTheme, onWorkspaceStyle, onRefresh, onInstallAgentProvider, onAuthenticateAgentProvider, onSaveImageProvider, onDeleteImageProvider, onTestImageProvider, onClose }: {
    providers: ProviderStatus[]; defaultProvider: string; generationProvider: string; theme: string; workspaceStyle: StylePresetId; customStyles?: CustomArtStyle[];
    onDefaultProvider: (provider: string) => void | Promise<void>; onGenerationProvider: (provider: string) => void | Promise<void>; onTheme: (theme: string) => void; onWorkspaceStyle: (style: StylePresetId) => void | Promise<void>;
    onRefresh: () => void | Promise<void>; onInstallAgentProvider: (providerId: string) => Promise<void>; onAuthenticateAgentProvider: (providerId: string) => Promise<void>; onSaveImageProvider: (input: ImageProviderInput) => Promise<void>; onDeleteImageProvider: (id: string) => Promise<void>;
    onTestImageProvider: (input: ImageProviderInput) => Promise<ProviderConnectionTest>; onClose: () => void;
  } = $props();

  let section = $state("providers");
  let editingProvider = $state<ImageProviderInput>();
  let savingProvider = $state(false);
  let testingProvider = $state(false);
  let testResult = $state<{ kind: "success" | "error"; detail: string }>();
  let refreshing = $state(false);
  let ollamaSettings = $state<OllamaSettings>({ baseUrl: "http://127.0.0.1:11434", model: "", hasToken: false });
  let ollamaDraft = $state<OllamaSettingsInput>({ baseUrl: "http://127.0.0.1:11434", model: "", bearerToken: "" });
  let ollamaModels = $state<OllamaModel[]>([]);
  let ollamaBusy = $state(false);
  let ollamaTesting = $state(false);
  let ollamaResult = $state<{ kind: "success" | "error"; detail: string }>();
  let mfluxSettings = $state<MfluxSettings>({ repository: "mflux-community/z-image-turbo-mflux-q8", revision: "4430e72e37bf2bc7bc889a42d306ae1b8d3b22de", runtimeVersion: "python3.11;mflux==0.19.1;mlx==0.32.0", runtimeReady: false, supportedHost: false, status: "unsupported", detail: "MFLUX status is loading.", cachePath: "" });
  let mfluxDraft = $state<MfluxSettingsInput>({ repository: "mflux-community/z-image-turbo-mflux-q8", revision: "4430e72e37bf2bc7bc889a42d306ae1b8d3b22de" });
  let mfluxBusy = $state(false);
  let mfluxSetupId = $state<string>();
  let mfluxProgress = $state(0);
  let mfluxResult = $state<{ kind: "success" | "error"; detail: string }>();
  let installingProvider = $state<string | null>(null);
  let authenticatingProvider = $state<string | null>(null);
  const sections = [{ id: "providers", name: "Providers", icon: Bot }, { id: "image-providers", name: "Image generation", icon: Image }, { id: "generation", name: "Project art", icon: Palette }, { id: "appearance", name: "Appearance", icon: Settings }, { id: "general", name: "General", icon: ShieldCheck }];
  const agents = $derived(providers.filter((item) => item.kind === "agent"));
  const nativeImageOrder = ["cursor-image", "imagegen", "antigravity-image"] as const;
  const nativeImageProviders = $derived(nativeImageOrder.map((id) => providers.find((item) => item.id === id)).filter((item): item is ProviderStatus => Boolean(item)));
  const customProviders = $derived(providers.filter((item) => item.kind === "image" && item.configurable && item.id !== "midjourney"));

  function statusLabel(status: string) { return ({ ready: "Ready", detected: "Detected", needs_auth: "Sign in required", not_installed: "Not installed", needs_codex: "Needs Codex", needs_cursor: "Needs Cursor", needs_antigravity: "Needs Antigravity", unsupported: "Unsupported host", needs_setup: "Needs setup", offline: "Offline", unavailable: "Needs setup" } as Record<string, string>)[status] ?? "Unavailable"; }
  function nativeImageTitle(id: string, name: string) { return id === "imagegen" ? "Codex ImageGen" : name; }
  function nativeImageHint(id: string) {
    if (id === "cursor-image") return "No API key. Sign in to Cursor CLI under Providers, then choose Cursor Image in a Cursor chat.";
    if (id === "imagegen") return "No API key. Sign in to Codex CLI under Providers, then choose Codex ImageGen in a Codex chat.";
    if (id === "antigravity-image") return "No API key. Sign in to Antigravity CLI under Providers, then choose Antigravity Image in an Antigravity chat.";
    return "";
  }
  function installableAgent(id: string) { return id === "cursor" || id === "antigravity"; }
  function installLabel(id: string) { return id === "cursor" ? "Install Cursor CLI" : "Install Antigravity CLI"; }
  function signInLabel(id: string) { return id === "cursor" ? "Sign in to Cursor" : "Sign in to Antigravity"; }
  function editProvider(provider?: ProviderStatus, preset?: "grok" | "gemini" | "midjourney") {
    testResult = undefined;
    if (provider) editingProvider = { id: provider.id, name: provider.name, providerType: provider.id === "grok-image" ? "grok" : "openai-compatible", baseUrl: provider.baseUrl ?? "", apiKey: "", model: provider.model ?? "" };
    else if (preset === "grok") editingProvider = { id: "grok-image", name: "Grok Image", providerType: "grok", baseUrl: "https://api.x.ai/v1", apiKey: "", model: "grok-imagine-image-quality" };
    else if (preset === "gemini") editingProvider = { id: "gemini-image", name: "Gemini Image", providerType: "openai-compatible", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", apiKey: "", model: "gemini-2.5-flash-image" };
    else if (preset === "midjourney") editingProvider = { id: "midjourney-gateway", name: "Midjourney", providerType: "openai-compatible", baseUrl: "https://", apiKey: "", model: "" };
    else editingProvider = { id: `custom-${Date.now().toString(36)}`, name: "", providerType: "openai-compatible", baseUrl: "https://", apiKey: "", model: "" };
    section = "image-providers";
  }
  async function refresh() { refreshing = true; try { await onRefresh(); } finally { refreshing = false; } }
  async function installProvider(providerId: string) { installingProvider = providerId; try { await onInstallAgentProvider(providerId); } finally { installingProvider = null; } }
  async function authenticateProvider(providerId: string) { authenticatingProvider = providerId; try { await onAuthenticateAgentProvider(providerId); } finally { authenticatingProvider = null; } }
  async function saveProvider() { if (!editingProvider) return; savingProvider = true; testResult = undefined; try { await onSaveImageProvider(editingProvider); editingProvider = undefined; } finally { savingProvider = false; } }
  async function testProvider() { if (!editingProvider) return; testingProvider = true; testResult = undefined; try { const result = await onTestImageProvider(editingProvider); testResult = { kind: "success", detail: result.detail }; } catch (error) { testResult = { kind: "error", detail: error instanceof Error ? error.message : String(error) }; } finally { testingProvider = false; } }
  async function removeProvider(id: string) { await onDeleteImageProvider(id); if (editingProvider?.id === id) editingProvider = undefined; }
  async function loadOllama() {
    try {
      const settings = await api.getOllamaSettings();
      ollamaSettings = settings;
      ollamaDraft = { baseUrl: settings.baseUrl, model: settings.model ?? "", bearerToken: "" };
      ollamaModels = await api.refreshOllamaModels();
      ollamaResult = undefined;
    } catch (error) {
      ollamaResult = { kind: "error", detail: error instanceof Error ? error.message : String(error) };
    }
  }
  async function refreshOllama() {
    ollamaBusy = true;
    ollamaResult = undefined;
    try { ollamaModels = await api.refreshOllamaModels(); }
    catch (error) { ollamaResult = { kind: "error", detail: error instanceof Error ? error.message : String(error) }; }
    finally { ollamaBusy = false; }
  }
  async function saveOllama() {
    ollamaBusy = true;
    ollamaResult = undefined;
    try {
      ollamaSettings = await api.saveOllamaSettings(ollamaDraft);
      ollamaDraft = { ...ollamaDraft, baseUrl: ollamaSettings.baseUrl, model: ollamaSettings.model ?? "", bearerToken: "" };
      ollamaModels = await api.refreshOllamaModels();
      await onRefresh();
      ollamaResult = { kind: "success", detail: "Ollama settings saved." };
    } catch (error) { ollamaResult = { kind: "error", detail: error instanceof Error ? error.message : String(error) }; }
    finally { ollamaBusy = false; }
  }
  async function testOllama() {
    ollamaTesting = true;
    ollamaResult = undefined;
    try { const result = await api.testOllamaConnection(ollamaDraft); ollamaResult = { kind: "success", detail: result.detail }; }
    catch (error) { ollamaResult = { kind: "error", detail: error instanceof Error ? error.message : String(error) }; }
    finally { ollamaTesting = false; }
  }
  async function loadMflux() {
    try {
      mfluxSettings = await api.getMfluxSettings();
      mfluxDraft = { repository: mfluxSettings.repository, revision: mfluxSettings.revision };
    } catch (error) {
      mfluxResult = { kind: "error", detail: error instanceof Error ? error.message : String(error) };
    }
  }
  async function saveMflux() {
    mfluxBusy = true;
    mfluxResult = undefined;
    try {
      mfluxSettings = await api.saveMfluxSettings(mfluxDraft);
      await onRefresh();
      mfluxResult = { kind: "success", detail: "MFLUX checkpoint settings saved." };
    } catch (error) {
      mfluxResult = { kind: "error", detail: error instanceof Error ? error.message : String(error) };
    } finally {
      mfluxBusy = false;
    }
  }
  async function startMfluxSetup() {
    mfluxBusy = true;
    mfluxResult = undefined;
    mfluxProgress = 0;
    try {
      mfluxSetupId = await api.startMfluxSetup();
    } catch (error) {
      mfluxBusy = false;
      mfluxResult = { kind: "error", detail: error instanceof Error ? error.message : String(error) };
    }
  }
  async function cancelMfluxSetup() {
    if (!mfluxSetupId) return;
    try { await api.cancelMfluxSetup(mfluxSetupId); }
    catch (error) { mfluxResult = { kind: "error", detail: error instanceof Error ? error.message : String(error) }; }
  }
  onMount(() => {
    void loadOllama();
    void loadMflux();
    const unlistenPromise = listen<MfluxSetupEvent>("mflux-setup-event", ({ payload }) => {
      if (payload.setupId !== mfluxSetupId) return;
      mfluxProgress = payload.progress;
      if (payload.eventType === "completed" || payload.eventType === "failed" || payload.eventType === "cancelled") {
        mfluxBusy = false;
        mfluxSetupId = undefined;
        mfluxResult = { kind: payload.eventType === "completed" ? "success" : "error", detail: payload.message };
        void loadMflux();
        if (payload.eventType === "completed") void onRefresh();
      }
    });
    return () => { unlistenPromise.then(unlisten => unlisten()); };
  });
</script>

<div class="backdrop" role="presentation" onclick={(event) => event.target === event.currentTarget && onClose()}>
  <div class="modal" role="dialog" aria-modal="true" aria-label="Settings">
    <header class="titlebar"><div class="brand"><div class="brand-mark"><Settings size={15}/></div><div><strong>Settings</strong><span>Sprite Studio</span></div></div><button class="icon-button" aria-label="Close settings" onclick={onClose}><X size={17}/></button></header>
    <div class="layout">
      <aside><p>Workspace</p>{#each sections as item}{@const Icon = item.icon}<button class:active={section === item.id} onclick={() => section = item.id}><Icon size={15}/><span>{item.name}</span>{#if section === item.id}<i></i>{/if}</button>{/each}<div class="privacy"><ShieldCheck size={15}/><div><strong>Local-first</strong><span>Keys stay in this app's local database and are never shown after save.</span></div></div></aside>
      <main>
        {#if section === "providers"}
          <div class="heading"><div><span class="eyebrow">AI runtime</span><h2>Providers</h2><p>Choose the CLI used for new chats. Existing chats keep their original provider.</p></div><button class="secondary" disabled={refreshing} onclick={refresh}><RefreshCw size={13} class={refreshing ? "spin" : ""}/>{refreshing ? "Detecting…" : "Detect again"}</button></div>
          <div class="section-label"><Terminal size={13}/><span>Agent CLIs</span></div>

          <div class="provider-grid">{#each agents as provider}<article class:chosen={defaultProvider === provider.id}><div class="provider-top"><div class="provider-icon"><Bot size={18}/></div><span class:ready={provider.status === "ready"} class:detected={provider.status === "detected"}>{#if provider.status === "ready"}<CheckCircle2 size={12}/>{:else}<CircleSlash2 size={12}/>{/if}{statusLabel(provider.status)}</span></div><h3>{provider.name}</h3><p>{provider.detail}</p>{#if provider.modes.length}<small>{provider.modes.length} model{provider.modes.length === 1 ? "" : "s"} reported by CLI</small>{/if}{#if provider.executable}<code title={provider.executable}>{provider.executable}</code>{/if}<button class="choose" disabled={!(["ready", "detected"].includes(provider.status)) || defaultProvider === provider.id} onclick={() => onDefaultProvider(provider.id)}>{defaultProvider === provider.id ? "Default for new chats" : "Use for new chats"}</button></article>{/each}</div>
          <div class="handoff-card"><div><strong>Generation handoff</strong><p>Ollama plans slash-command requests, then this provider performs workspace and file actions.</p></div><select value={generationProvider} onchange={(event) => onGenerationProvider(event.currentTarget.value)}>{#each agents.filter(item => item.id !== "ollama") as provider}<option value={provider.id}>{provider.name}</option>{/each}</select></div>
          <div class="ollama-card"><div class="heading-inline"><div><strong>Ollama endpoint and model</strong><p>Use an existing local or remote Ollama tag. Sprite Studio never creates, pulls, or changes models.</p></div><button class="secondary" disabled={ollamaBusy || ollamaTesting} onclick={refreshOllama}><RefreshCw size={13} class={ollamaBusy ? "spin" : ""}/>{ollamaBusy ? "Refreshing…" : "Refresh models"}</button></div><div class="form-grid"><label class="wide">Base URL<input bind:value={ollamaDraft.baseUrl} placeholder="http://127.0.0.1:11434" inputmode="url" autocomplete="url"/><small>Local HTTP is allowed; remote endpoints must use HTTPS.</small></label><label>Model tag<select bind:value={ollamaDraft.model}><option value="">Choose an installed model</option>{#each ollamaModels as model}<option value={model.model}>{model.name}{model.vision ? " · vision" : ""}</option>{/each}</select><small>Tags are discovered from /api/tags. You may enter a custom tag below.</small></label><label>Custom model tag<input bind:value={ollamaDraft.model} placeholder="llama3.2:latest" autocomplete="off"/></label><label class="wide">Bearer token<input type="password" bind:value={ollamaDraft.bearerToken} placeholder={ollamaSettings.hasToken ? "Saved — leave blank to keep it" : "Optional for remote endpoints"} autocomplete="new-password"/><small>Stored locally and never included in error messages.</small></label></div>{#if ollamaResult}<div class="test-result" class:success={ollamaResult.kind === "success"} class:error={ollamaResult.kind === "error"}>{ollamaResult.detail}</div>{/if}<footer class="ollama-footer"><span>{ollamaModels.length ? `${ollamaModels.length} installed model${ollamaModels.length === 1 ? "" : "s"} found` : "No model list loaded"}</span><div><button class="secondary" disabled={ollamaTesting || ollamaBusy} onclick={testOllama}><PlugZap size={13}/>{ollamaTesting ? "Testing…" : "Test connection"}</button><button class="primary" disabled={ollamaTesting || ollamaBusy} onclick={saveOllama}>{ollamaBusy ? "Saving…" : "Save Ollama settings"}</button></div></footer></div>
          {#each agents.filter(item => installableAgent(item.id) && ["not_installed", "needs_auth"].includes(item.status)) as provider}
            <div class="provider-action-row">
              <span>{provider.name}: {statusLabel(provider.status)}</span>
              {#if provider.status === "not_installed"}
                <button class="install" disabled={installingProvider === provider.id || refreshing} onclick={() => installProvider(provider.id)}><Download size={13} class={installingProvider === provider.id ? "spin" : ""}/>{installingProvider === provider.id ? "Installing…" : installLabel(provider.id)}</button>
              {:else}
                <button class="install" disabled={authenticatingProvider === provider.id || refreshing} onclick={() => authenticateProvider(provider.id)}><LogIn size={13} class={authenticatingProvider === provider.id ? "spin" : ""}/>{authenticatingProvider === provider.id ? "Opening sign-in…" : signInLabel(provider.id)}</button>
              {/if}
            </div>
          {/each}
          <!--
          <div class="provider-grid">{#each agents as provider}<article class:chosen={defaultProvider === provider.id}><div class="provider-top"><div class="provider-icon"><Bot size={18}/></div><span class:ready={provider.status === "ready"} class:detected={provider.status === "detected"}>{#if provider.status === "ready"}<CheckCircle2 size={12}/>{:else}<CircleSlash2 size={12}/>{/if}{statusLabel(provider.status)}</span></div><h3>{provider.name}</h3><p>{provider.detail}</p>{#if provider.modes.length}<small>{provider.modes.length} model{provider.modes.length === 1 ? "" : "s"} reported by CLI</small>{/if}{#if provider.executable}<code title={provider.executable}>{provider.executable}</code>{/if}{#if installableAgent(provider.id) && provider.status === "not_installed"}<button class="install" disabled={installingProvider === provider.id || refreshing} onclick={() => installProvider(provider.id)}><Download size={13} class={installingProvider === provider.id ? "spin" : ""}/>{installingProvider === provider.id ? "Installing…" : installLabel(provider.id)}</button>{:else if installableAgent(provider.id) && provider.status === "needs_auth"}<button class="install" disabled={authenticatingProvider === provider.id || refreshing} onclick={() => authenticateProvider(provider.id)}><LogIn size={13} class={authenticatingProvider === provider.id ? "spin" : ""}/>{authenticatingProvider === provider.id ? "Opening sign-in…" : signInLabel(provider.id)}</button>{/if}<button class="choose" disabled={!(["ready", "detected"].includes(provider.status)) || defaultProvider === provider.id} onclick={() => onDefaultProvider(provider.id)}>{defaultProvider === provider.id ? "Default for new chats" : "Use for new chats"}</button></article>{/each}</div>
          -->
        {:else if section === "image-providers"}
          <div class="heading"><div><span class="eyebrow">Image models</span><h2>Image generation</h2><p>Cursor Image, Codex ImageGen, and Antigravity Image follow the signed-in CLI and need no API key. Grok, Gemini, and authorized endpoints are configured below.</p></div><button class="secondary" onclick={() => editProvider()}><Plus size={13}/>Custom image API</button></div>
          <div class="mflux-card">
            <div class="heading-inline"><div><strong>MFLUX · Z-Image Turbo</strong><p>Optional Apple-Silicon runtime for local master, animation-frame, text-to-image, and one-reference image-to-image generation.</p></div><span class:ready={mfluxSettings.status === "ready"} class:unsupported={mfluxSettings.status === "unsupported"}>{statusLabel(mfluxSettings.status)}</span></div>
            <div class="form-grid">
              <label>Hugging Face repository<input bind:value={mfluxDraft.repository} disabled={mfluxBusy} autocomplete="off"/><small>Use a repository that contains a compatible Z-Image Turbo checkpoint.</small></label>
              <label>Immutable revision<input bind:value={mfluxDraft.revision} disabled={mfluxBusy} autocomplete="off"/><small>Exactly 40 hexadecimal commit characters.</small></label>
            </div>
            <div class="mflux-meta"><span>Runtime: {mfluxSettings.runtimeVersion}</span><code title={mfluxSettings.cachePath}>{mfluxSettings.cachePath || "Managed application cache"}</code></div>
            {#if mfluxSetupId}<div class="mflux-progress"><div><span>Setup progress</span><strong>{Math.round(mfluxProgress * 100)}%</strong></div><progress max="1" value={mfluxProgress}></progress></div>{/if}
            <p class="mflux-detail">{mfluxSettings.detail}</p>
            {#if mfluxResult}<div class="test-result" class:success={mfluxResult.kind === "success"} class:error={mfluxResult.kind === "error"}>{mfluxResult.detail}</div>{/if}
            <footer class="ollama-footer"><span>{mfluxSettings.supportedHost ? "Nothing is installed inside project folders." : "Unavailable on this host."}</span><div><button class="secondary" disabled={mfluxBusy} onclick={saveMflux}>{mfluxBusy && !mfluxSetupId ? "Saving…" : "Save checkpoint"}</button>{#if mfluxSetupId}<button class="secondary" onclick={cancelMfluxSetup}>Cancel setup</button>{:else}<button class="primary" disabled={!mfluxSettings.supportedHost || mfluxBusy} onclick={startMfluxSetup}>{mfluxSettings.runtimeReady ? "Repair runtime" : "Install runtime"}</button>{/if}</div></footer>
          </div>
          <div class="section-label"><Terminal size={13}/><span>CLI image models</span></div>
          <div class="image-provider-cards">
            {#each nativeImageProviders as provider}
              <article class="native-image-card">
                <div class="provider-top"><div class="provider-icon"><Image size={17}/></div><span class:ready={provider.status === "ready"} class:detected={provider.status === "detected"}>{#if provider.status === "ready"}<CheckCircle2 size={12}/>{:else}<CircleSlash2 size={12}/>{/if}{statusLabel(provider.status)}</span></div>
                <h3>{nativeImageTitle(provider.id, provider.name)}</h3>
                <p>{nativeImageHint(provider.id)}</p>
                <button class="choose" type="button" onclick={() => section = "providers"}>{provider.status === "ready" ? "View in Providers" : "Set up in Providers"}</button>
              </article>
            {/each}
          </div>
          <div class="section-label"><PlugZap size={13}/><span>Bring-your-own APIs</span></div>
          <div class="image-provider-cards">
            <button class="image-provider-card" onclick={() => editProvider(providers.find(item => item.id === "grok-image"), "grok")}><div class="provider-icon"><Image size={17}/></div><span><strong>Grok Image</strong><small>Native xAI image API · works with Grok, Gemini, Claude, or Codex chats</small></span><ArrowRight size={15}/></button>
            <button class="image-provider-card" onclick={() => editProvider(providers.find(item => item.id === "gemini-image"), "gemini")}><div class="provider-icon"><Image size={17}/></div><span><strong>Gemini Image</strong><small>Native Gemini image API · model prefilled as Gemini 2.5 Flash Image</small></span><ArrowRight size={15}/></button>
            <button class="image-provider-card" onclick={() => editProvider(undefined, "midjourney")}><div class="provider-icon"><Image size={17}/></div><span><strong>Midjourney gateway</strong><small>Only for an endpoint you are authorized to use; no web or Discord automation</small></span><ArrowRight size={15}/></button>
          </div>
          <div class="section-label"><PlugZap size={13}/><span>Configured image APIs</span></div>
          <div class="custom-layout">
            <div class="saved-list">{#if customProviders.length === 0}<div class="empty"><PlugZap size={22}/><strong>No image APIs configured</strong><p>Choose Grok, Gemini, or add an authorized compatible endpoint above.</p></div>{/if}{#each customProviders as provider}<button class="saved-provider" class:active={editingProvider?.id === provider.id} onclick={() => editProvider(provider)}><div class="provider-icon"><PlugZap size={16}/></div><span><strong>{provider.name}</strong><small>{provider.model || "Model not set"}</small></span><i class:ready={provider.status === "ready"}>{statusLabel(provider.status)}</i></button>{/each}<div class="gateway-note"><strong>How it works</strong><p>Choose an image API in the chat’s generation menu. The chat provider can be different from the image provider.</p></div></div>
            {#if editingProvider}
              <form class="provider-form" onsubmit={(event) => { event.preventDefault(); saveProvider(); }}><div class="form-heading"><div><span class="eyebrow">Endpoint details</span><h3>{editingProvider.name || "New provider"}</h3></div><button type="button" class="icon-button" aria-label="Close provider editor" onclick={() => editingProvider = undefined}><X size={15}/></button></div>
                <div class="form-grid"><label>Display name<input bind:value={editingProvider.name} placeholder="My image provider" autocomplete="off"/></label><label>Provider ID<input bind:value={editingProvider.id} placeholder="my-provider" disabled={providers.some((item) => item.kind === "image" && item.id === editingProvider?.id && item.hasApiKey)}/><small>Letters, numbers, dashes, or underscores.</small></label><label class="wide">Base URL<input bind:value={editingProvider.baseUrl} placeholder="https://api.example.com/v1" inputmode="url" autocomplete="url"/></label><label>Model ID<input bind:value={editingProvider.model} placeholder="image-model-id" autocomplete="off"/></label><label>Provider type<select bind:value={editingProvider.providerType}><option value="openai-compatible">OpenAI compatible</option><option value="grok">Grok Imagine</option></select></label><label class="wide">API key<input type="password" bind:value={editingProvider.apiKey} placeholder={providers.find((item) => item.kind === "image" && item.id === editingProvider?.id)?.hasApiKey ? "Saved — leave blank to keep existing key" : "Required"} autocomplete="new-password"/><small>Stored locally. Test and error messages never include the key.</small></label></div>
                {#if testResult}<div class="test-result" class:success={testResult.kind === "success"} class:error={testResult.kind === "error"}>{testResult.detail}</div>{/if}
                <footer>{#if editingProvider.id !== "grok-image" && providers.some((item) => item.kind === "image" && item.id === editingProvider?.id && item.id !== "midjourney")}<button type="button" class="danger" onclick={() => removeProvider(editingProvider!.id)}><Trash2 size={13}/>Delete</button>{/if}<div><button type="button" class="secondary" disabled={testingProvider || savingProvider} onclick={testProvider}><PlugZap size={13}/>{testingProvider ? "Testing…" : "Test connection"}</button><button type="submit" class="primary" disabled={savingProvider || testingProvider}>{savingProvider ? "Saving…" : "Save provider"}</button></div></footer>
              </form>
            {:else}<div class="form-placeholder"><PlugZap size={24}/><strong>Select or add a provider</strong><p>The form includes only the fields needed by the supported image-generation contract.</p></div>{/if}
          </div>
        {:else if section === "generation"}
          <div class="heading"><div><span class="eyebrow">Visual direction</span><h2>Project art</h2><p>This becomes the default art direction for every chat in this project.</p></div></div><StylePicker value={workspaceStyle} {customStyles} onChange={(value) => { if (value !== "inherit") return onWorkspaceStyle(value); }}/><div class="note"><strong>Chats can override this default.</strong><p>Open the art control in a chat header to choose a direction for only that conversation.</p></div>
        {:else if section === "appearance"}
          <div class="heading"><div><span class="eyebrow">Interface</span><h2>Appearance</h2><p>Choose how Sprite Studio looks on this computer.</p></div></div><label class="setting-row"><span><strong>Theme</strong><small>Applied immediately and saved locally.</small></span><select value={theme} onchange={(event) => onTheme(event.currentTarget.value)}><option value="system">System</option><option value="dark">Dark</option><option value="light">Light</option></select></label>
        {:else}<div class="heading"><div><span class="eyebrow">About your data</span><h2>General</h2><p>Sprite Studio is local-first by design.</p></div></div><div class="note"><strong>Bring your own AI. Own your assets.</strong><p>Images stay in your project folder and application metadata stays in SQLite. Removing a project from the app does not delete its files.</p></div>{/if}
      </main>
    </div>
  </div>
</div>

<style>
  .backdrop{position:fixed;inset:0;z-index:70;display:grid;place-items:center;padding:22px;background:#050605c9;backdrop-filter:blur(7px)}.modal{width:min(1040px,calc(100vw - 44px));height:min(750px,calc(100vh - 44px));display:flex;flex-direction:column;overflow:hidden;border:1px solid #46483f;border-radius:15px;background:var(--bg);box-shadow:0 38px 120px #000d,0 0 0 1px #b4c36b0a;color:var(--text)}
  .titlebar{height:58px;min-height:58px;display:flex;align-items:center;justify-content:space-between;padding:0 14px 0 18px;border-bottom:1px solid var(--border);background:linear-gradient(180deg,#ffffff05,transparent)}.brand{display:flex;align-items:center;gap:10px}.brand-mark{width:30px;height:30px;display:grid;place-items:center;border:1px solid #a7b66638;border-radius:8px;background:#a7b66610;color:var(--accent)}.brand strong,.brand span{display:block}.brand strong{font-size:13px}.brand span{margin-top:2px;font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint)}.icon-button{width:31px;height:31px;display:grid;place-items:center;padding:0;border:0;border-radius:7px;background:transparent;color:var(--faint);cursor:pointer}.icon-button:hover{background:var(--surface-hover);color:var(--text)}
  .layout{flex:1;min-height:0;display:grid;grid-template-columns:214px minmax(0,1fr)}aside{position:relative;display:flex;flex-direction:column;padding:20px 12px;border-right:1px solid var(--border);background:var(--sidebar)}aside>p{margin:0 9px 10px;font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--faint)}aside>button{position:relative;width:100%;height:39px;display:grid;grid-template-columns:18px 1fr 4px;align-items:center;gap:9px;padding:0 10px;border:0;border-radius:7px;background:transparent;color:var(--muted);font:inherit;font-size:12px;text-align:left;cursor:pointer}aside>button:hover{background:var(--surface-hover);color:var(--text)}aside>button.active{background:#a7b66612;color:var(--text)}aside>button.active :global(svg){color:var(--accent)}aside>button i{width:4px;height:17px;border-radius:4px;background:var(--accent)}.privacy{display:grid;grid-template-columns:18px 1fr;gap:9px;margin:auto 5px 2px;padding:12px;border:1px solid var(--border);border-radius:9px;background:#0c0e0c80;color:var(--accent)}.privacy strong,.privacy span{display:block}.privacy strong{font-size:10px;color:var(--muted)}.privacy span{margin-top:5px;font-size:9px;line-height:1.45;color:var(--faint)}
  main{min-width:0;overflow:auto;padding:38px 42px 48px;background:radial-gradient(circle at 90% 0,#a7b66609,transparent 33%)}.heading{display:flex;align-items:flex-start;justify-content:space-between;gap:28px;padding-bottom:23px;border-bottom:1px solid var(--border)}.eyebrow{display:block;margin-bottom:7px;font-size:9px;font-weight:750;letter-spacing:.15em;text-transform:uppercase;color:var(--accent)}h2{margin:0;font-size:25px;letter-spacing:-.035em}h3{margin:0}.heading p{max-width:550px;margin:8px 0 0;font-size:12px;line-height:1.55;color:var(--muted)}button{font-family:inherit}.secondary,.primary,.choose,.install{height:32px;display:inline-flex;align-items:center;justify-content:center;gap:6px;padding:0 10px;border:1px solid var(--border-strong);border-radius:7px;background:var(--surface);color:var(--muted);font-size:10px;cursor:pointer;white-space:nowrap}.secondary:hover,.choose:hover,.install:hover{border-color:#a7b66655;color:var(--text)}button:disabled{opacity:.45;cursor:not-allowed}.primary{border-color:var(--accent);background:var(--accent);color:#171912;font-weight:700}.install{width:100%;margin-top:13px;border-color:#a7b66655;background:#a7b66610;color:var(--accent);font-weight:650}:global(.spin){animation:spin .9s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
  .section-label{display:flex;align-items:center;gap:7px;margin:22px 0 10px;font-size:9px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:var(--faint)}.provider-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.provider-grid article{min-width:0;padding:15px;border:1px solid var(--border);border-radius:10px;background:var(--sidebar)}.provider-grid article.chosen{border-color:#a7b66655;box-shadow:inset 0 0 0 1px #a7b66612}.provider-top{display:flex;align-items:flex-start;justify-content:space-between}.provider-icon{width:34px;height:34px;display:grid;place-items:center;border:1px solid var(--border);border-radius:8px;background:var(--surface);color:var(--muted)}.provider-top>span{display:inline-flex;align-items:center;gap:5px;padding:4px 6px;border-radius:5px;background:#ffffff05;color:var(--faint);font-size:9px}.provider-top>span.ready{background:#7da46b16;color:#8eb57a}.provider-top>span.detected{background:#b6a56816;color:#c2ad68}.provider-grid h3{margin-top:13px;font-size:13px}.provider-grid p{min-height:47px;margin:6px 0 8px;font-size:10px;line-height:1.5;color:var(--muted)}.provider-grid small{display:block;font-size:9px;color:var(--faint)}code{display:block;overflow:hidden;margin-top:7px;color:var(--faint);font:9px ui-monospace,SFMono-Regular,monospace;text-overflow:ellipsis;white-space:nowrap}.choose{width:100%;margin-top:13px;background:#101210}.chosen .choose{border-color:transparent;background:#a7b66610;color:var(--accent)}
  .section-label+.image-provider-cards{margin-top:0}.image-provider-cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:20px}.native-image-card{display:flex;flex-direction:column;padding:12px;border:1px solid var(--border);border-radius:10px;background:var(--sidebar)}.native-image-card h3{margin-top:10px;font-size:12px}.native-image-card p{flex:1;margin:6px 0 0;font-size:9px;line-height:1.45;color:var(--muted)}.image-provider-card{display:grid;grid-template-columns:34px minmax(0,1fr) 16px;align-items:center;gap:10px;min-height:82px;padding:12px;border:1px solid var(--border);border-radius:10px;background:var(--sidebar);color:var(--text);font:inherit;text-align:left;cursor:pointer}.image-provider-card:hover{border-color:#a7b66655;background:var(--surface-hover)}.image-provider-card strong,.image-provider-card small{display:block}.image-provider-card strong{font-size:11px}.image-provider-card small{margin-top:5px;font-size:9px;line-height:1.4;color:var(--faint)}.image-provider-card :global(svg:last-child){color:var(--accent)}
  .custom-layout{display:grid;grid-template-columns:235px minmax(0,1fr);min-height:435px;margin-top:18px;border:1px solid var(--border);border-radius:11px;overflow:hidden;background:var(--sidebar)}.saved-list{padding:9px;border-right:1px solid var(--border);background:#090b09}.saved-provider{width:100%;display:grid;grid-template-columns:32px minmax(0,1fr) auto;align-items:center;gap:9px;padding:9px;border:0;border-radius:7px;background:transparent;color:var(--text);text-align:left;cursor:pointer}.saved-provider:hover,.saved-provider.active{background:var(--selected)}.saved-provider .provider-icon{width:32px;height:32px}.saved-provider strong,.saved-provider small{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.saved-provider strong{font-size:10px}.saved-provider small{margin-top:4px;font-size:9px;color:var(--faint)}.saved-provider i{font-size:8px;font-style:normal;color:var(--faint)}.saved-provider i.ready{color:#8eb57a}.empty{padding:28px 12px;text-align:center;color:var(--faint)}.empty strong{display:block;margin-top:10px;font-size:11px;color:var(--muted)}.empty p{font-size:9px;line-height:1.5}.gateway-note{margin:16px 4px 4px;padding:11px;border:1px solid var(--border);border-radius:8px;background:var(--surface)}.gateway-note strong{font-size:10px}.gateway-note p{margin:5px 0 9px;font-size:9px;line-height:1.5;color:var(--faint)}
  .provider-form{padding:25px 27px}.form-heading{display:flex;align-items:flex-start;justify-content:space-between}.form-heading h3{font-size:17px}.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:15px 12px;margin-top:22px}.form-grid label{font-size:10px;color:var(--muted)}.form-grid label.wide{grid-column:1/-1}.form-grid input,.form-grid select{width:100%;height:36px;display:block;margin-top:6px;padding:0 10px;border:1px solid var(--border-strong);border-radius:7px;outline:0;background:var(--bg);color:var(--text);font:inherit;font-size:11px}.form-grid input:focus,.form-grid select:focus{border-color:var(--accent);box-shadow:0 0 0 3px #a7b66612}.form-grid input:disabled{opacity:.6}.form-grid small{display:block;margin-top:5px;font-size:8px;line-height:1.4;color:var(--faint)}.provider-form footer{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:20px;padding-top:16px;border-top:1px solid var(--border)}.provider-form footer>div{display:flex;gap:7px;margin-left:auto}.danger{height:32px;display:flex;align-items:center;gap:6px;padding:0 9px;border:1px solid #6f4141;border-radius:7px;background:transparent;color:#cf7a75;font:inherit;font-size:10px;cursor:pointer}.test-result{margin-top:13px;padding:9px 10px;border:1px solid var(--border);border-radius:6px;font-size:10px;line-height:1.45;color:var(--muted)}.test-result.success{border-color:#597551;color:#91b785;background:#7193660d}.test-result.error{border-color:#754848;color:#d88a84;background:#9a55550d}.form-placeholder{display:grid;place-content:center;justify-items:center;padding:40px;color:var(--faint);text-align:center}.form-placeholder strong{margin-top:12px;font-size:12px;color:var(--muted)}.form-placeholder p{max-width:300px;margin:7px 0;font-size:10px;line-height:1.5}
  .note{margin-top:22px;padding:16px;border:1px solid var(--border);border-radius:9px;background:var(--surface)}.note strong,.setting-row strong{font-size:12px}.note p{margin:6px 0 0;font-size:11px;line-height:1.55;color:var(--muted)}.setting-row{height:70px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border)}.setting-row strong,.setting-row small{display:block}.setting-row small{margin-top:4px;font-size:10px;color:var(--faint)}.setting-row select{height:32px;padding:0 8px;border:1px solid var(--border);border-radius:6px;background:var(--surface);color:var(--text);font:inherit;font-size:11px}@media(max-width:820px){.modal{width:calc(100vw - 24px);height:calc(100vh - 24px)}.layout{grid-template-columns:170px minmax(0,1fr)}main{padding:28px 24px}.provider-grid,.image-provider-cards{grid-template-columns:1fr}.custom-layout{grid-template-columns:1fr}.saved-list{border-right:0;border-bottom:1px solid var(--border)}.provider-grid p{min-height:0}}
  .handoff-card,.ollama-card{margin-top:15px;padding:15px;border:1px solid var(--border);border-radius:10px;background:var(--surface)}.handoff-card{display:flex;align-items:center;justify-content:space-between;gap:16px}.handoff-card strong,.ollama-card strong{font-size:12px}.handoff-card p,.ollama-card p{margin:5px 0 0;font-size:10px;line-height:1.45;color:var(--muted)}.handoff-card select{height:32px;min-width:170px;padding:0 8px;border:1px solid var(--border-strong);border-radius:6px;background:var(--bg);color:var(--text);font:inherit;font-size:10px}.heading-inline{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.ollama-card .form-grid{margin-top:16px}.ollama-footer{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:16px;padding-top:13px;border-top:1px solid var(--border);font-size:9px;color:var(--faint)}.ollama-footer>div{display:flex;gap:7px}
  .mflux-card{margin-top:15px;padding:15px;border:1px solid #a7b66638;border-radius:10px;background:linear-gradient(135deg,#a7b6660d,transparent 65%),var(--surface)}.mflux-card strong{font-size:12px}.mflux-card .heading-inline>span{display:inline-flex;padding:4px 6px;border-radius:5px;background:#ffffff05;color:var(--faint);font-size:9px}.mflux-card .heading-inline>span.ready{background:#7da46b16;color:#8eb57a}.mflux-card .heading-inline>span.unsupported{background:#9a55550d;color:#d88a84}.mflux-card .form-grid{margin-top:16px}.mflux-meta{display:flex;justify-content:space-between;gap:12px;margin-top:13px;color:var(--faint);font-size:9px}.mflux-meta code{max-width:52%;margin:0}.mflux-detail{margin:10px 0 0;font-size:10px;line-height:1.45;color:var(--muted)}.mflux-progress{margin-top:13px}.mflux-progress>div{display:flex;justify-content:space-between;font-size:9px;color:var(--faint)}.mflux-progress strong{color:var(--text)}.mflux-progress progress{width:100%;height:5px;margin-top:7px;accent-color:var(--accent)}
</style>
