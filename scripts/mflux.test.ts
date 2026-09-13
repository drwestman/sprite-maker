import { describe, expect, test } from "bun:test";
import { mergeMfluxProvider, mfluxProviderStatus } from "../src/lib/mflux";
import type { MfluxSettings, ProviderStatus } from "../src/lib/types";

const settings = (overrides: Partial<MfluxSettings> = {}): MfluxSettings => ({
  repository: "mflux-community/z-image-turbo-mflux-q8",
  revision: "4430e72e37bf2bc7bc889a42d306ae1b8d3b22de",
  runtimeVersion: "python3.11;mflux==0.19.1;mlx==0.32.0",
  runtimeReady: false,
  supportedHost: false,
  status: "unsupported",
  detail: "MFLUX requires Apple Silicon macOS.",
  cachePath: "/managed/mflux",
  ...overrides,
});

const provider = (id: string): ProviderStatus => ({
  id,
  name: id,
  kind: "image",
  installed: true,
  status: "ready",
  detail: "",
  modes: [],
  capabilities: {
    textInput: true,
    imageInput: false,
    multipleImageInput: false,
    imageEditing: false,
    masks: false,
    transparency: false,
    structuredOutput: false,
    videoAnimation: false,
    imageToImage: false,
    maximumReferenceImages: 0,
  },
  configurable: false,
  hasApiKey: false,
});

describe("MFLUX provider readiness", () => {
  test("keeps unsupported or incomplete runtimes unavailable", () => {
    expect(mfluxProviderStatus(settings()).status).toBe("unsupported");
    expect(mfluxProviderStatus(settings({ supportedHost: true, status: "needs_setup" })).installed).toBe(false);
    expect(mfluxProviderStatus(settings({ supportedHost: true, runtimeReady: true, status: "ready" })).installed).toBe(true);
  });

  test("merges one MFLUX entry without duplicating a stale provider", () => {
    const merged = mergeMfluxProvider([provider("mflux"), provider("grok-image")], settings({ supportedHost: true, status: "needs_setup" }));
    expect(merged.filter(item => item.id === "mflux")).toHaveLength(1);
    expect(merged.find(item => item.id === "mflux")?.status).toBe("needs_setup");
    expect(merged.find(item => item.id === "grok-image")).toBeDefined();
  });
});
