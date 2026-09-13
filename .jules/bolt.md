# Bolt's Journal

## 2026-09-13 - O(1) Map Lookup Optimization for Artifact Cards and Generation Inference
**Learning:** Reactivity and data derivation in Svelte components rendering lists of assets (such as sprite artifact cards and pack cards) as well as animation frame resolution routines can hit O(M * N) complexity when resolving frames via `Array.prototype.find`. Mapping assets by ID or relative path using `Map` converts frame resolution to O(M + N).
**Action:** When mapping array elements against another collection by identifier, construct a `Map` or derived Map first to maintain O(1) lookup speed per item.
