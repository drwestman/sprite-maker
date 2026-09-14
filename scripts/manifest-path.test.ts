import { describe, expect, test } from "bun:test";
import { buildAssetManifestMap, findAssetByManifestPath, manifestPathsMatch, normalizeManifestPath } from "../src/lib/manifest-path";
import type { Asset } from "../src/lib/types";

const asset = (relativePath: string): Asset => ({
  id: relativePath,
  name: relativePath,
  workspaceId: "ws",
  path: `/tmp/${relativePath}`,
  relativePath,
  category: "characters",
  format: "png",
  width: 64,
  height: 64,
  fileSize: 100,
  hasAlpha: true,
  createdAt: "now",
});

describe("manifest path normalization", () => {
  test("normalizes slashes and leading ./", () => {
    expect(normalizeManifestPath(".\\assets\\frame-01.png")).toBe("assets/frame-01.png");
    expect(normalizeManifestPath("./assets/frame-01.png")).toBe("assets/frame-01.png");
    expect(manifestPathsMatch("assets/a.png", ".\\assets\\a.png")).toBe(true);
  });

  test("finds assets across path separator styles", () => {
    const library = [asset("assets/characters/hero.png")];
    expect(findAssetByManifestPath(library, "assets\\characters\\hero.png")?.id).toBe("assets/characters/hero.png");
  });

  test("builds asset manifest map for O(1) lookups with normalized keys", () => {
    const library = [asset("assets/characters/hero.png"), asset("assets/props/sword.png")];
    const map = buildAssetManifestMap(library);
    expect(map.get("assets/characters/hero.png")?.id).toBe("assets/characters/hero.png");
    expect(map.get(normalizeManifestPath(".\\assets\\props\\sword.png"))?.id).toBe("assets/props/sword.png");
  });

  test("Given leading-dot and Windows paths, when building the map, then keys are canonical", () => {
    const map = buildAssetManifestMap([
      asset("./assets/characters/hero.png"),
      asset(".\\assets\\props\\sword.png"),
    ]);

    expect([...map.keys()]).toEqual(["assets/characters/hero.png", "assets/props/sword.png"]);
    expect(map.get("assets/characters/hero.png")?.id).toBe("./assets/characters/hero.png");
    expect(map.get("assets/props/sword.png")?.id).toBe(".\\assets\\props\\sword.png");
  });

  test("Given duplicate normalized paths, when building the map, then the later asset wins", () => {
    const first = asset("assets/characters/hero.png");
    const second = asset(".\\assets\\characters\\hero.png");
    const map = buildAssetManifestMap([first, second]);

    expect(map.size).toBe(1);
    expect(map.get("assets/characters/hero.png")?.id).toBe(second.id);
    expect(findAssetByManifestPath([first, second], "assets/characters/hero.png")?.id).toBe(second.id);
  });

  test("Given no assets, when building the map, then it is empty", () => {
    expect(buildAssetManifestMap([])).toEqual(new Map());
  });
});
