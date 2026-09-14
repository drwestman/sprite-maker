import { describe, expect, test } from "bun:test";
import { GENERATION_PRESETS, normalizeGenerationProfile, selectProviderModel } from "../src/lib/generation-profiles";
import { defaultImageProviderId } from "../src/lib/generation-profiles";

describe("generation profile defaults", () => {
  test("uses a 128 by 128 mid-quality canvas by default", () => {
    const profile = normalizeGenerationProfile(null);
    expect(profile.quality).toBe("mid");
    expect([profile.width, profile.height]).toEqual([128, 128]);
    expect(profile.profileVersion).toBe(9);
    expect(profile.imageInputMode).toBe("text-to-image");
    expect(profile.imageStrength).toBe(0.4);
  });

  test("upgrades saved preset profiles while preserving custom dimensions", () => {
    const legacy = normalizeGenerationProfile({ profileVersion: 7, quality: "mid", width: 64, height: 64 });
    expect([legacy.width, legacy.height]).toEqual([128, 128]);

    const custom = normalizeGenerationProfile({ profileVersion: 7, quality: "custom", width: 96, height: 80 });
    expect([custom.width, custom.height]).toEqual([96, 80]);
    expect(GENERATION_PRESETS.mid.width).toBe(128);
  });

  test("selects a saved, provider-default, or first available model in that order", () => {
    const modes = [
      { id: "llama3.2:latest", label: "Llama", description: "", defaultReasoningEffort: "", reasoningEfforts: [] },
      { id: "qwen2.5:latest", label: "Qwen", description: "", defaultReasoningEffort: "", reasoningEfforts: [] },
    ];

    expect(selectProviderModel("  saved:model  ", "default:model", modes)).toBe("saved:model");
    expect(selectProviderModel("", "default:model", modes)).toBe("default:model");
    expect(selectProviderModel("", "", modes)).toBe("llama3.2:latest");
    expect(selectProviderModel("", "", [])).toBe("");
  });

  test("maps cursor chats to native cursor image generation", () => {
    expect(defaultImageProviderId("cursor")).toBe("cursor-image");
    expect(defaultImageProviderId("codex")).toBe("imagegen");
    expect(defaultImageProviderId("claude")).toBe("provider-native");
    expect(normalizeGenerationProfile({ imageProviderId: "imagegen" }, [], "cursor").imageProviderId).toBe("cursor-image");
    expect(normalizeGenerationProfile({ imageProviderId: "grok-image" }, [], "cursor").imageProviderId).toBe("grok-image");
  });

  test("maps antigravity chats to native antigravity image generation", () => {
    expect(defaultImageProviderId("antigravity")).toBe("antigravity-image");
    expect(normalizeGenerationProfile({ imageProviderId: "imagegen" }, [], "antigravity").imageProviderId).toBe("antigravity-image");
    expect(normalizeGenerationProfile({ imageProviderId: "grok-image" }, [], "antigravity").imageProviderId).toBe("grok-image");
  });

  test("normalizes MFLUX image input settings and strength", () => {
    const profile = normalizeGenerationProfile({ imageProviderId: "mflux", imageInputMode: "image-to-image", imageStrength: 2 });
    expect(profile.imageProviderId).toBe("mflux");
    expect(profile.imageInputMode).toBe("image-to-image");
    expect(profile.imageStrength).toBe(1);
    expect(normalizeGenerationProfile({ imageStrength: -1 }).imageStrength).toBe(0);
  });
});
