import type { Asset } from "$lib/types";

/** Normalize workspace-relative paths for cross-platform manifest matching. */
export function normalizeManifestPath(path: string): string {
  return path.replace(/\\/g, "/").replace(/^\.\//, "");
}

export function manifestPathsMatch(left: string, right: string): boolean {
  return normalizeManifestPath(left) === normalizeManifestPath(right);
}

/** Build an asset map keyed by normalized relative path for O(1) lookups during manifest reconciliation. */
export function buildAssetManifestMap(assets: Asset[]): Map<string, Asset> {
  const map = new Map<string, Asset>();
  for (const asset of assets) {
    map.set(normalizeManifestPath(asset.relativePath), asset);
  }
  return map;
}

/** Resolve an asset by manifest path, ignoring slash style differences. */
export function findAssetByManifestPath(assets: Asset[], path: string): Asset | undefined {
  const normalized = normalizeManifestPath(path);
  return assets.find(asset => normalizeManifestPath(asset.relativePath) === normalized);
}
