import {
    type Connection,
    type GridState,
    type Machine,
    MachineType,
    Orientation,
} from './types';
import { createGrid, getMachinePorts, placeMachine } from './grid';
import { applyBeltPath, findBeltPath } from './pathfinder';

export interface BenchmarkScenario {
    name: string;
    gridSize: number;
    machines: Machine[];
    connections: Connection[];
}

type RandomFn = () => number;

interface RandomScenarioSpec {
    seed: number;
    machineCount: number;
    edgeDensity: number;
    name: string;
    gridSize?: number;
}

const GRID_SIZE = 50;
const ORIENTATIONS = [Orientation.NORTH, Orientation.EAST, Orientation.SOUTH, Orientation.WEST];

export function createAdversarialScenarios(): BenchmarkScenario[] {
    return [
        buildDenseCrossbarScenario(),
        buildChokepointCorridorScenario(),
        buildMixedSizeCongestionScenario(),
        buildCyclicChordConflictScenario(),
    ];
}

export function createRandomDataset(dataset: 'tuning' | 'holdout'): BenchmarkScenario[] {
    const specs: RandomScenarioSpec[] = dataset === 'tuning'
        ? [
            { seed: 11, machineCount: 12, edgeDensity: 1.8, name: 'tuning-11' },
            { seed: 23, machineCount: 14, edgeDensity: 1.9, name: 'tuning-23' },
            { seed: 37, machineCount: 10, edgeDensity: 2.0, name: 'tuning-37' },
            { seed: 53, machineCount: 16, edgeDensity: 1.7, name: 'tuning-53' },
            { seed: 67, machineCount: 13, edgeDensity: 2.0, name: 'tuning-67' },
            { seed: 79, machineCount: 15, edgeDensity: 1.8, name: 'tuning-79' },
            { seed: 83, machineCount: 17, edgeDensity: 2.1, name: 'tuning-83-stress', gridSize: 42 },
            { seed: 97, machineCount: 18, edgeDensity: 2.2, name: 'tuning-97-stress', gridSize: 42 },
        ]
        : [
            { seed: 101, machineCount: 12, edgeDensity: 1.8, name: 'holdout-101' },
            { seed: 131, machineCount: 15, edgeDensity: 1.8, name: 'holdout-131' },
            { seed: 167, machineCount: 11, edgeDensity: 2.0, name: 'holdout-167' },
            { seed: 197, machineCount: 16, edgeDensity: 1.7, name: 'holdout-197' },
            { seed: 223, machineCount: 13, edgeDensity: 1.9, name: 'holdout-223' },
            { seed: 251, machineCount: 14, edgeDensity: 1.8, name: 'holdout-251' },
            { seed: 281, machineCount: 17, edgeDensity: 2.1, name: 'holdout-281-stress', gridSize: 42 },
            { seed: 313, machineCount: 18, edgeDensity: 2.2, name: 'holdout-313-stress', gridSize: 42 },
        ];

    return specs.map((spec) => generateRoutableRandomScenario(spec));
}

export function createGridFromScenario(scenario: BenchmarkScenario): GridState {
    const grid = createGrid(scenario.gridSize, scenario.gridSize);

    for (const machine of scenario.machines) {
        if (!placeMachine(grid, { ...machine })) {
            throw new Error(`Invalid machine placement in scenario "${scenario.name}" (${machine.id})`);
        }
    }

    for (const connection of scenario.connections) {
        grid.connections.set(connection.id, { ...connection });
    }

    for (const connection of scenario.connections) {
        const srcMachine = grid.machines.get(connection.sourceMachineId);
        const tgtMachine = grid.machines.get(connection.targetMachineId);
        if (!srcMachine || !tgtMachine) {
            throw new Error(`Invalid connection endpoints in scenario "${scenario.name}" (${connection.id})`);
        }
        const srcPorts = getMachinePorts(srcMachine).outputs;
        const tgtPorts = getMachinePorts(tgtMachine).inputs;

        let bestPath: ReturnType<typeof findBeltPath> = null;
        let bestSourceIndex = connection.sourcePortIndex;
        let bestTargetIndex = connection.targetPortIndex;

        for (let si = 0; si < srcPorts.length; si++) {
            const srcPort = srcPorts[si];
            if (!srcPort) continue;
            for (let ti = 0; ti < tgtPorts.length; ti++) {
                const tgtPort = tgtPorts[ti];
                if (!tgtPort) continue;
                const candidate = findBeltPath(grid, srcPort, tgtPort, connection.id);
                if (!candidate) continue;
                if (!bestPath || candidate.segments.length < bestPath.segments.length) {
                    bestPath = candidate;
                    bestSourceIndex = si;
                    bestTargetIndex = ti;
                }
            }
        }

        if (!bestPath) {
            throw new Error(`Unroutable connection in scenario "${scenario.name}" (${connection.id})`);
        }

        const storedConnection = grid.connections.get(connection.id);
        if (!storedConnection) {
            throw new Error(`Missing connection in scenario "${scenario.name}" (${connection.id})`);
        }
        storedConnection.sourcePortIndex = bestSourceIndex;
        storedConnection.targetPortIndex = bestTargetIndex;
        applyBeltPath(grid, bestPath);
    }

    return grid;
}

export function summarize(values: number[]): { mean: number; p50: number; p90: number } {
    if (values.length === 0) return { mean: 0, p50: 0, p90: 0 };
    const sorted = values.slice().sort((a, b) => a - b);
    const pick = (quantile: number): number => {
        const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * quantile)));
        return sorted[idx];
    };
    const mean = sorted.reduce((sum, v) => sum + v, 0) / sorted.length;
    return { mean, p50: pick(0.5), p90: pick(0.9) };
}

export function createSeededRandom(seed: number): RandomFn {
    let state = (seed >>> 0) || 0x6d2b79f5;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 0x100000000;
    };
}

function buildDenseCrossbarScenario(): BenchmarkScenario {
    const sources = Array.from(
        { length: 3 },
        (_, i) => makeMachine(`crossbar_s${i}`, MachineType.M5x3, 4 + i * 16, 4, Orientation.SOUTH),
    );
    const sinks = Array.from(
        { length: 3 },
        (_, i) => makeMachine(`crossbar_t${i}`, MachineType.M5x3, 4 + i * 16, 38, Orientation.NORTH),
    );
    const machines = [...sources, ...sinks];
    const connections: Connection[] = [];
    let id = 0;
    for (let s = 0; s < sources.length; s++) {
        for (let t = 0; t < sinks.length; t++) {
            connections.push({
                id: `crossbar_c${id++}`,
                sourceMachineId: sources[s].id,
                sourcePortIndex: (s + t * 2) % 5,
                targetMachineId: sinks[t].id,
                targetPortIndex: (t + s * 2) % 5,
            });
        }
    }

    return {
        name: 'dense crossbar',
        gridSize: GRID_SIZE,
        machines,
        connections,
    };
}

function buildChokepointCorridorScenario(): BenchmarkScenario {
    const left = Array.from({ length: 4 }, (_, i) => makeMachine(`corr_l${i}`, MachineType.M3x3, 3, 6 + i * 9, Orientation.EAST));
    const right = Array.from({ length: 4 }, (_, i) => makeMachine(`corr_r${i}`, MachineType.M3x3, 43, 6 + i * 9, Orientation.WEST));
    const blockers: Machine[] = [
        makeMachine('corr_b0', MachineType.M5x5, 20, 0, Orientation.NORTH),
        makeMachine('corr_b1', MachineType.M5x5, 20, 8, Orientation.NORTH),
        makeMachine('corr_b2', MachineType.M5x5, 20, 16, Orientation.NORTH),
        makeMachine('corr_b3', MachineType.M5x5, 20, 31, Orientation.NORTH),
        makeMachine('corr_b4', MachineType.M5x5, 27, 0, Orientation.NORTH),
        makeMachine('corr_b5', MachineType.M5x5, 27, 8, Orientation.NORTH),
        makeMachine('corr_b6', MachineType.M5x5, 27, 16, Orientation.NORTH),
        makeMachine('corr_b7', MachineType.M5x5, 27, 31, Orientation.NORTH),
    ];
    const machines = [...left, ...right, ...blockers];
    const connections: Connection[] = [];
    let id = 0;
    for (let i = 0; i < left.length; i++) {
        connections.push({
            id: `corr_c${id++}`,
            sourceMachineId: left[i].id,
            sourcePortIndex: i % 3,
            targetMachineId: right[(i + 1) % right.length].id,
            targetPortIndex: (i + 2) % 3,
        });
        connections.push({
            id: `corr_c${id++}`,
            sourceMachineId: left[i].id,
            sourcePortIndex: (i + 1) % 3,
            targetMachineId: right[i].id,
            targetPortIndex: i % 3,
        });
    }

    return {
        name: 'chokepoint corridor',
        gridSize: GRID_SIZE,
        machines,
        connections,
    };
}

function buildMixedSizeCongestionScenario(): BenchmarkScenario {
    const machines: Machine[] = [
        makeMachine('mix_a', MachineType.M5x5, 6, 6, Orientation.SOUTH),
        makeMachine('mix_b', MachineType.M5x3, 16, 7, Orientation.EAST),
        makeMachine('mix_c', MachineType.M3x3, 25, 8, Orientation.WEST),
        makeMachine('mix_d', MachineType.M5x5, 33, 6, Orientation.NORTH),
        makeMachine('mix_e', MachineType.M3x3, 9, 20, Orientation.EAST),
        makeMachine('mix_f', MachineType.M5x3, 19, 20, Orientation.SOUTH),
        makeMachine('mix_g', MachineType.M5x5, 30, 20, Orientation.WEST),
        makeMachine('mix_h', MachineType.M3x3, 12, 33, Orientation.NORTH),
        makeMachine('mix_i', MachineType.M5x3, 23, 34, Orientation.WEST),
        makeMachine('mix_j', MachineType.M3x3, 36, 34, Orientation.SOUTH),
    ];

    const pairs: Array<[number, number]> = [
        [0, 1], [0, 4], [1, 2], [1, 5], [2, 3], [2, 6], [3, 6],
        [4, 5], [4, 7], [5, 6], [5, 8], [6, 9], [7, 8], [8, 9],
        [7, 5], [8, 6], [9, 3], [6, 2],
    ];
    const connections = pairs.map(([srcIdx, tgtIdx], idx) => ({
        id: `mix_c${idx}`,
        sourceMachineId: machines[srcIdx].id,
        sourcePortIndex: idx % portCountForType(machines[srcIdx].type),
        targetMachineId: machines[tgtIdx].id,
        targetPortIndex: (idx + 1) % portCountForType(machines[tgtIdx].type),
    }));

    return {
        name: 'mixed-size congestion',
        gridSize: GRID_SIZE,
        machines,
        connections,
    };
}

function buildCyclicChordConflictScenario(): BenchmarkScenario {
    const machines = Array.from({ length: 10 }, (_, i) => {
        const ringPos = [
            { x: 2, y: 2 }, { x: 14, y: 3 }, { x: 26, y: 2 }, { x: 38, y: 3 }, { x: 42, y: 15 },
            { x: 36, y: 30 }, { x: 24, y: 38 }, { x: 12, y: 35 }, { x: 3, y: 24 }, { x: 6, y: 12 },
        ][i];
        return makeMachine(
            `cycle_m${i}`,
            MachineType.M3x3,
            ringPos.x,
            ringPos.y,
            ORIENTATIONS[i % ORIENTATIONS.length],
        );
    });

    const connections: Connection[] = [];
    let id = 0;
    for (let i = 0; i < machines.length; i++) {
        const next = (i + 1) % machines.length;
        connections.push({
            id: `cycle_c${id++}`,
            sourceMachineId: machines[i].id,
            sourcePortIndex: i % 3,
            targetMachineId: machines[next].id,
            targetPortIndex: (i + 2) % 3,
        });
    }
    const chords: Array<[number, number]> = [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]];
    for (const [src, tgt] of chords) {
        connections.push({
            id: `cycle_c${id++}`,
            sourceMachineId: machines[src].id,
            sourcePortIndex: (src + 1) % 3,
            targetMachineId: machines[tgt].id,
            targetPortIndex: (tgt + 2) % 3,
        });
    }

    return {
        name: 'cyclic + long chords with port conflicts',
        gridSize: GRID_SIZE,
        machines,
        connections,
    };
}

function generateRoutableRandomScenario(spec: RandomScenarioSpec): BenchmarkScenario {
    for (let attempt = 0; attempt < 80; attempt++) {
        const candidate = generateRandomScenario({ ...spec, seed: spec.seed + attempt * 997 });
        try {
            createGridFromScenario(candidate);
            return candidate;
        } catch {
            // Retry with deterministic offset until routable.
        }
    }
    throw new Error(`Failed to generate routable scenario for ${spec.name}`);
}

function generateRandomScenario(spec: RandomScenarioSpec): BenchmarkScenario {
    const random = createSeededRandom(spec.seed);
    const machines: Machine[] = [];
    const layers = 3;
    const layerAssignments: number[][] = Array.from({ length: layers }, () => []);
    for (let i = 0; i < spec.machineCount; i++) {
        layerAssignments[i % layers].push(i);
    }

    const layerOf = new Map<number, number>();
    for (let layer = 0; layer < layerAssignments.length; layer++) {
        const ids = layerAssignments[layer];
        for (let slot = 0; slot < ids.length; slot++) {
            const index = ids[slot];
            layerOf.set(index, layer);
            machines.push({
                id: `${spec.name}_m${index}`,
                type: pickMachineType(random),
                x: 2 + slot * 7,
                y: 2 + layer * 15,
                orientation: Orientation.NORTH,
            });
        }
    }

    machines.sort((a, b) => parseMachineIndex(a.id) - parseMachineIndex(b.id));

    const targetEdges = Math.max(spec.machineCount - 1, Math.round(spec.machineCount * spec.edgeDensity));
    const connections: Connection[] = [];
    const seenPairs = new Set<string>();
    const outUsage = new Map<number, number>();
    const inUsage = new Map<number, number>();

    for (let i = 0; i < machines.length - 1; i++) {
        const srcLayer = layerOf.get(i) ?? 0;
        const tgtLayer = layerOf.get(i + 1) ?? 0;
        if (srcLayer < tgtLayer && hasPortCapacity(machines, outUsage, inUsage, i, i + 1)) {
            addConnection(machines, connections, seenPairs, outUsage, inUsage, i, i + 1, `${spec.name}_base${i}`);
        }
    }

    let guard = 0;
    while (connections.length < targetEdges && guard < 5000) {
        guard++;
        const src = Math.floor(random() * machines.length);
        const tgt = Math.floor(random() * machines.length);
        if (src === tgt) continue;
        const srcLayer = layerOf.get(src) ?? 0;
        const tgtLayer = layerOf.get(tgt) ?? 0;
        if (srcLayer >= tgtLayer) continue;
        const key = `${src}->${tgt}`;
        if (seenPairs.has(key)) continue;
        if (!hasPortCapacity(machines, outUsage, inUsage, src, tgt)) continue;
        addConnection(machines, connections, seenPairs, outUsage, inUsage, src, tgt, `${spec.name}_extra${connections.length}`);
    }

    return {
        name: `random ${spec.name} (n=${spec.machineCount}, d=${spec.edgeDensity.toFixed(1)})`,
        gridSize: spec.gridSize ?? GRID_SIZE,
        machines,
        connections,
    };
}

function addConnection(
    machines: Machine[],
    connections: Connection[],
    seenPairs: Set<string>,
    outUsage: Map<number, number>,
    inUsage: Map<number, number>,
    srcIndex: number,
    tgtIndex: number,
    id: string,
): void {
    const src = machines[srcIndex];
    const tgt = machines[tgtIndex];
    const srcPort = outUsage.get(srcIndex) ?? 0;
    const tgtPort = inUsage.get(tgtIndex) ?? 0;
    connections.push({
        id,
        sourceMachineId: src.id,
        sourcePortIndex: srcPort,
        targetMachineId: tgt.id,
        targetPortIndex: tgtPort,
    });
    seenPairs.add(`${srcIndex}->${tgtIndex}`);
    outUsage.set(srcIndex, srcPort + 1);
    inUsage.set(tgtIndex, tgtPort + 1);
}

function hasPortCapacity(
    machines: Machine[],
    outUsage: Map<number, number>,
    inUsage: Map<number, number>,
    srcIndex: number,
    tgtIndex: number,
): boolean {
    const srcLimit = portCountForType(machines[srcIndex].type);
    const tgtLimit = portCountForType(machines[tgtIndex].type);
    return (outUsage.get(srcIndex) ?? 0) < srcLimit && (inUsage.get(tgtIndex) ?? 0) < tgtLimit;
}

function parseMachineIndex(id: string): number {
    const marker = id.lastIndexOf('_m');
    return marker >= 0 ? Number(id.slice(marker + 2)) : 0;
}

function makeMachine(
    id: string,
    type: MachineType,
    x: number,
    y: number,
    orientation: Orientation,
): Machine {
    return { id, type, x, y, orientation };
}

function pickMachineType(random: RandomFn): MachineType {
    const roll = random();
    if (roll < 0.45) return MachineType.M3x3;
    if (roll < 0.75) return MachineType.M5x3;
    return MachineType.M5x5;
}

function portCountForType(type: MachineType): number {
    switch (type) {
        case MachineType.M3x3:
            return 3;
        case MachineType.M5x3:
        case MachineType.M5x5:
            return 5;
    }
}
