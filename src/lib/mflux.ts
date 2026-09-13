import type { MfluxSettings, ProviderStatus, SpriteSlashCommand } from "$lib/types";

const MFLUX_CAPABILITIES = {
  textInput: true,
  imageInput: true,
  multipleImageInput: false,
  imageEditing: true,
  masks: false,
  transparency: false,
  structuredOutput: false,
  videoAnimation: false,
  imageToImage: true,
  maximumReferenceImages: 1,
};

export function mfluxProviderStatus(settings: MfluxSettings): ProviderStatus {
  return {
    id: "mflux",
    name: "MFLUX Z-Image Turbo",
    kind: "image",
    installed: settings.runtimeReady,
    status: settings.status,
    detail: settings.detail,
    modes: [{
      id: "z-image-turbo",
      label: "Z-Image Turbo",
      description: `Pinned checkpoint ${settings.repository}@${settings.revision.slice(0, 12)}`,
      defaultReasoningEffort: "",
      reasoningEfforts: [],
    }],
    capabilities: MFLUX_CAPABILITIES,
    configurable: false,
    hasApiKey: false,
    model: "z-image-turbo",
  };
}

export function mergeMfluxProvider(providers: ProviderStatus[], settings: MfluxSettings): ProviderStatus[] {
  return [...providers.filter(provider => provider.id !== "mflux"), mfluxProviderStatus(settings)];
}

const ANIMATION_PROMPT_TERMS = [
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
];

/** Mirrors the backend's animation inference for MFLUX source-master selection. */
export function mfluxAnimationRequested(command: SpriteSlashCommand | undefined, prompt: string): boolean {
  if (command === "animate") return true;
  if (command) return false;
  const lower = prompt.toLowerCase();
  return ANIMATION_PROMPT_TERMS.some(term => lower.includes(term));
}

export function selectMfluxReferenceId(focusedReferenceId: string | undefined, referenceIds: string[]): string | undefined {
  const focused = focusedReferenceId && referenceIds.includes(focusedReferenceId) ? focusedReferenceId : undefined;
  return focused ?? (referenceIds.length === 1 ? referenceIds[0] : undefined);
}

export function mfluxReferenceRequired(
  command: SpriteSlashCommand | undefined,
  prompt: string,
  hasAnimationMaster: boolean,
  hasSelectedReference: boolean,
): boolean {
  if (hasSelectedReference) return false;
  return !mfluxAnimationRequested(command, prompt) || !hasAnimationMaster;
}
