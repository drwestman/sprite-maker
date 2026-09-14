import { describe, expect, test } from "bun:test";
import {
  mergeMfluxProvider,
  mfluxAnimationRequested,
  mfluxProviderStatus,
  mfluxReferenceRequired,
  selectMfluxReferenceId,
} from "../src/lib/mflux";
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

  test("given a prompt when animation inference runs then it matches backend semantics", () => {
    expect(mfluxAnimationRequested("animate", "make a walk cycle")).toBe(true);
    expect(mfluxAnimationRequested(undefined, "create a looping idle animation")).toBe(true);
    expect(mfluxAnimationRequested(undefined, "create a static treasure chest")).toBe(false);
    expect(mfluxAnimationRequested("sprite", "create an animated treasure chest")).toBe(false);
  });

  test("given active references when selecting input then focus wins or the sole reference is used", () => {
    expect(selectMfluxReferenceId("focused", ["focused", "other"])).toBe("focused");
    expect(selectMfluxReferenceId("missing", ["only"])).toBe("only");
    expect(selectMfluxReferenceId(undefined, ["first", "second"])).toBeUndefined();
  });

  test("given static image-to-image with a selected asset when no reference is selected then it requires a reference", () => {
    expect(mfluxReferenceRequired(undefined, "create a static treasure chest", true, false)).toBe(true);
    expect(mfluxReferenceRequired("animate", "make a walk cycle", true, false)).toBe(false);
    expect(mfluxReferenceRequired("animate", "make a walk cycle", false, false)).toBe(true);
    expect(mfluxReferenceRequired(undefined, "create a static treasure chest", true, true)).toBe(false);
  });
});
