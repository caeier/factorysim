# Factory Grid Simulator

A browser-based tool for placing machines on a grid, connecting them with conveyor belts, and automatically optimizing the layout for minimal space and belt usage.

**[Try it live](https://caeier.github.io/factorysim/)**

---

## What it does

You define a factory layout by placing machines of different sizes (3x3, 5x3, 5x5) onto a 50x50 grid. Each machine has input ports and output ports. You connect outputs to inputs, and the system routes conveyor belts between them using A\* pathfinding.

The optimizer rearranges everything — machine positions, orientations, port assignments, belt routing — to minimize the total score (a weighted sum of belt count, bounding box area, and corner count).

## Optimizer

The layout optimizer is a multi-phase simulated annealing pipeline:

- **Phase 0** — Seed generation using greedy placement, topology-aware layering, pattern matching, and two-layer exhaustive search. The best seed is selected by routed score.
- **Phase 1** — Fast SA using Manhattan distance as a proxy score. Supports restarts and elite archive seeding.
- **Phase 2** — Fine-tuned SA with full A\* belt routing for accurate scoring.
- **Phase 3** — Port assignment optimization via brute-force permutation search.
- **Phase 4** — Layout compaction and orientation polish with optional SA interleaving.

Adaptive operator selection adjusts move probabilities based on recent improvement history. Move operators include neighbor-directed shifts, port-facing jumps, position swaps, cluster destroy-repair, and critical net focus moves.

**Search Deeper** runs the optimizer in a continuous loop, persisting an elite archive across cycles and auto-stopping on plateau.

## Controls

| Action | Key / Input |
|---|---|
| Place machine | `P` + click |
| Select / drag | `S` + click |
| Connect ports | `C` + click output then input |
| Delete machine | `D` + click |
| Rotate | `R` |
| Machine sizes | `1` / `2` / `3` |
| Optimize | `O` |
| Deep search | `Shift+O` |
| Pan | Right-click drag |
| Zoom | Scroll wheel |
| Stop search | `Escape` |

Layouts can be exported and imported as JSON.

## Running locally

```
npm install
npm run dev
```

## Building

```
npm run build
```

Produces a single self-contained `dist/index.html` (all JS and CSS inlined).

## Stack

TypeScript, Vite, HTML Canvas. No frameworks, no dependencies beyond the build toolchain.
