import type { MfluxSettings, ProviderStatus } from "$lib/types";

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
