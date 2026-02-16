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
        buildTightPackingPinwheelScenario(),
        buildLongHaulRelayScenario(),
        buildPortPressureHubScenario(),
        buildAnchorInjectionScenario(),
        buildDualCorridorCongestionScenario(),
        buildMixedFootprintWeaveScenario(),
        buildCyclicBackflowScenario(),
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

function buildTightPackingPinwheelScenario(): BenchmarkScenario {
    const machines: Machine[] = [
        makeMachine('pin_core_a', MachineType.M5x5, 14, 14, Orientation.NORTH),
        makeMachine('pin_core_b', MachineType.M6x4, 22, 13, Orientation.WEST),
        makeMachine('pin_core_c', MachineType.M5x5, 28, 14, Orientation.SOUTH),
        makeMachine('pin_nw', MachineType.M3x3, 9, 8, Orientation.NORTH),
        makeMachine('pin_ne', MachineType.M3x3, 34, 8, Orientation.NORTH),
        makeMachine('pin_w', MachineType.M3x3, 8, 23, Orientation.WEST),
        makeMachine('pin_e', MachineType.M3x3, 36, 23, Orientation.EAST),
        makeMachine('pin_sw', MachineType.M3x3, 12, 33, Orientation.SOUTH),
        makeMachine('pin_se', MachineType.M3x3, 31, 33, Orientation.SOUTH),
    ];

    const pairs: Array<[number, number]> = [
        [3, 0], [4, 2], [5, 0], [6, 2],
        [0, 1], [2, 1], [1, 7], [1, 8],
        [0, 2], [2, 0], [3, 1], [4, 1], [5, 1], [6, 1],
    ];

    return {
        name: 'tight packing pinwheel',
        gridSize: GRID_SIZE,
        machines,
        connections: createConnectionsFromPairs('pin', machines, pairs),
    };
}

function buildLongHaulRelayScenario(): BenchmarkScenario {
    const machines: Machine[] = [
        makeMachine('haul_s0', MachineType.M3x3, 2, 5, Orientation.WEST),
        makeMachine('haul_s1', MachineType.M5x5, 2, 19, Orientation.WEST),
        makeMachine('haul_s2', MachineType.M6x4, 2, 35, Orientation.WEST),
        makeMachine('haul_t0', MachineType.M3x3, 43, 6, Orientation.WEST),
        makeMachine('haul_t1', MachineType.M5x5, 41, 20, Orientation.WEST),
        makeMachine('haul_t2', MachineType.M6x4, 42, 34, Orientation.WEST),
        makeMachine('haul_r0', MachineType.M3x3, 21, 10, Orientation.NORTH),
        makeMachine('haul_r1', MachineType.M5x5, 22, 22, Orientation.WEST),
        makeMachine('haul_r2', MachineType.M3x3, 21, 37, Orientation.SOUTH),
    ];

    const pairs: Array<[number, number]> = [
        [0, 6], [0, 7], [1, 6], [1, 7], [1, 8], [2, 7], [2, 8],
        [6, 3], [6, 4], [7, 3], [7, 4], [7, 5], [8, 4], [8, 5],
        [0, 3], [2, 5],
    ];

    return {
        name: 'long-haul relay bus',
        gridSize: GRID_SIZE,
        machines,
        connections: createConnectionsFromPairs('haul', machines, pairs),
    };
}

function buildPortPressureHubScenario(): BenchmarkScenario {
    const machines: Machine[] = [
        makeMachine('hub_in', MachineType.M6x4, 20, 18, Orientation.WEST),
        makeMachine('hub_out', MachineType.M6x4, 27, 18, Orientation.WEST),
        makeMachine('hub_src_nw', MachineType.M3x3, 14, 8, Orientation.NORTH),
        makeMachine('hub_src_n', MachineType.M3x3, 23, 8, Orientation.NORTH),
        makeMachine('hub_src_ne', MachineType.M3x3, 32, 8, Orientation.NORTH),
        makeMachine('hub_src_w', MachineType.M5x5, 10, 20, Orientation.WEST),
        makeMachine('hub_src_sw', MachineType.M3x3, 14, 34, Orientation.SOUTH),
        makeMachine('hub_sink_e', MachineType.M5x5, 36, 20, Orientation.WEST),
        makeMachine('hub_sink_se', MachineType.M3x3, 33, 35, Orientation.NORTH),
        makeMachine('hub_sink_nw', MachineType.M3x3, 20, 2, Orientation.SOUTH),
        makeMachine('hub_sink_far', MachineType.M3x3, 43, 9, Orientation.WEST),
    ];

    const pairs: Array<[number, number]> = [
        [2, 0], [3, 0], [4, 0], [5, 0], [6, 0],
        [0, 1], [0, 1], [0, 7],
        [1, 7], [1, 8], [1, 9], [1, 10], [1, 4], [1, 5],
    ];

    return {
        name: 'port-pressure twin hubs',
        gridSize: GRID_SIZE,
        machines,
        connections: createConnectionsFromPairs('hub', machines, pairs),
    };
}

function buildAnchorInjectionScenario(): BenchmarkScenario {
    const machines: Machine[] = [
        makeMachine('anc_l0', MachineType.M3x1, 5, 7, Orientation.EAST),
        makeMachine('anc_l1', MachineType.M3x1, 5, 20, Orientation.EAST),
        makeMachine('anc_l2', MachineType.M3x1, 5, 33, Orientation.EAST),
        makeMachine('anc_r0', MachineType.M3x1, 44, 10, Orientation.WEST),
        makeMachine('anc_r1', MachineType.M3x1, 44, 23, Orientation.WEST),
        makeMachine('anc_r2', MachineType.M3x1, 44, 36, Orientation.WEST),
        makeMachine('anc_p0', MachineType.M5x5, 14, 11, Orientation.WEST),
        makeMachine('anc_p1', MachineType.M6x4, 22, 10, Orientation.WEST),
        makeMachine('anc_p2', MachineType.M5x5, 14, 27, Orientation.WEST),
        makeMachine('anc_p3', MachineType.M6x4, 23, 29, Orientation.WEST),
        makeMachine('anc_sink', MachineType.M5x5, 31, 19, Orientation.WEST),
    ];

    const pairs: Array<[number, number]> = [
        [0, 6], [1, 6], [2, 8], [3, 7], [4, 9], [5, 9],
        [6, 7], [6, 8], [6, 9],
        [7, 10], [7, 8],
        [8, 9], [8, 10],
        [9, 10],
    ];

    return {
        name: 'anchor injection lanes',
        gridSize: GRID_SIZE,
        machines,
        connections: createConnectionsFromPairs('anc', machines, pairs),
    };
}

function buildDualCorridorCongestionScenario(): BenchmarkScenario {
    const left = Array.from(
        { length: 4 },
        (_, i) => makeMachine(`corr_l${i}`, MachineType.M3x3, 2, 5 + i * 10, Orientation.WEST),
    );
    const right = Array.from(
        { length: 4 },
        (_, i) => makeMachine(`corr_r${i}`, MachineType.M3x3, 45, 5 + i * 10, Orientation.WEST),
    );
    const blockers: Machine[] = [
        makeMachine('corr_b0', MachineType.M5x5, 18, 0, Orientation.NORTH),
        makeMachine('corr_b1', MachineType.M5x5, 18, 7, Orientation.NORTH),
        makeMachine('corr_b2', MachineType.M5x5, 18, 14, Orientation.NORTH),
        makeMachine('corr_b3', MachineType.M5x5, 18, 29, Orientation.NORTH),
        makeMachine('corr_b4', MachineType.M5x5, 18, 36, Orientation.NORTH),
        makeMachine('corr_b5', MachineType.M5x5, 27, 0, Orientation.NORTH),
        makeMachine('corr_b6', MachineType.M5x5, 27, 7, Orientation.NORTH),
        makeMachine('corr_b7', MachineType.M5x5, 27, 14, Orientation.NORTH),
        makeMachine('corr_b8', MachineType.M5x5, 27, 29, Orientation.NORTH),
        makeMachine('corr_b9', MachineType.M5x5, 27, 36, Orientation.NORTH),
    ];
    const machines = [...left, ...right, ...blockers];

    const pairs: Array<[number, number]> = [
        [0, 4], [0, 5], [0, 7],
        [1, 4], [1, 6],
        [2, 5], [2, 7],
        [3, 6], [3, 7], [3, 4],
    ];

    return {
        name: 'dual corridor congestion',
        gridSize: GRID_SIZE,
        machines,
        connections: createConnectionsFromPairs('corr', machines, pairs),
    };
}

function buildMixedFootprintWeaveScenario(): BenchmarkScenario {
    const machines: Machine[] = [
        makeMachine('weave_a', MachineType.M6x4, 6, 6, Orientation.WEST),
        makeMachine('weave_b', MachineType.M5x5, 16, 5, Orientation.SOUTH),
        makeMachine('weave_c', MachineType.M3x3, 29, 6, Orientation.NORTH),
        makeMachine('weave_d', MachineType.M5x5, 37, 8, Orientation.WEST),
        makeMachine('weave_e', MachineType.M3x3, 8, 23, Orientation.WEST),
        makeMachine('weave_f', MachineType.M6x4, 18, 22, Orientation.SOUTH),
        makeMachine('weave_g', MachineType.M5x5, 31, 23, Orientation.NORTH),
        makeMachine('weave_h', MachineType.M3x3, 42, 24, Orientation.EAST),
        makeMachine('weave_i', MachineType.M3x3, 14, 38, Orientation.NORTH),
        makeMachine('weave_j', MachineType.M6x4, 27, 37, Orientation.EAST),
        makeMachine('weave_k', MachineType.M5x5, 38, 37, Orientation.SOUTH),
    ];

    const pairs: Array<[number, number]> = [
        [0, 1], [0, 4], [0, 5],
        [1, 2], [1, 5], [1, 6],
        [2, 3], [2, 6], [2, 5],
        [3, 7],
        [4, 5], [4, 8],
        [5, 6], [5, 9], [5, 10],
        [6, 7], [6, 10],
        [8, 9], [9, 10],
    ];

    return {
        name: 'mixed-footprint weave',
        gridSize: GRID_SIZE,
        machines,
        connections: createConnectionsFromPairs('weave', machines, pairs),
    };
}

function buildCyclicBackflowScenario(): BenchmarkScenario {
    const machineSpecs: Array<{ id: string; type: MachineType; x: number; y: number }> = [
        { id: 'cycle_0', type: MachineType.M3x3, x: 4, y: 4 },
        { id: 'cycle_1', type: MachineType.M3x3, x: 16, y: 3 },
        { id: 'cycle_2', type: MachineType.M5x5, x: 30, y: 4 },
        { id: 'cycle_3', type: MachineType.M3x3, x: 42, y: 12 },
        { id: 'cycle_4', type: MachineType.M6x4, x: 40, y: 28 },
        { id: 'cycle_5', type: MachineType.M3x3, x: 27, y: 39 },
        { id: 'cycle_6', type: MachineType.M5x5, x: 13, y: 37 },
        { id: 'cycle_7', type: MachineType.M3x3, x: 3, y: 24 },
    ];
    const machines = machineSpecs.map((spec, index) =>
        makeMachine(spec.id, spec.type, spec.x, spec.y, ORIENTATIONS[index % ORIENTATIONS.length]),
    );

    const pairs: Array<[number, number]> = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [4, 5], [5, 6], [6, 7], [7, 0],
        [0, 4], [1, 5], [2, 6], [3, 7],
        [5, 1], [6, 2], [7, 3],
    ];

    return {
        name: 'cyclic backflow ring',
        gridSize: GRID_SIZE,
        machines,
        connections: createConnectionsFromPairs('cycle', machines, pairs),
    };
}

function createConnectionsFromPairs(
    prefix: string,
    machines: Machine[],
    pairs: Array<[number, number]>,
): Connection[] {
    const outUsage = new Map<number, number>();
    const inUsage = new Map<number, number>();

    return pairs.map(([srcIndex, tgtIndex], idx) => {
        const sourceMachine = machines[srcIndex];
        const targetMachine = machines[tgtIndex];
        if (!sourceMachine || !targetMachine) {
            throw new Error(`Invalid benchmark connection pair: ${srcIndex} -> ${tgtIndex}`);
        }
        const srcCount = outputPortCountForType(sourceMachine.type);
        const tgtCount = inputPortCountForType(targetMachine.type);
        if (srcCount <= 0 || tgtCount <= 0) {
            throw new Error(`Benchmark pair targets unsupported ports: ${sourceMachine.id} -> ${targetMachine.id}`);
        }
        const sourcePortIndex = (outUsage.get(srcIndex) ?? 0) % srcCount;
        const targetPortIndex = (inUsage.get(tgtIndex) ?? 0) % tgtCount;
        outUsage.set(srcIndex, (outUsage.get(srcIndex) ?? 0) + 1);
        inUsage.set(tgtIndex, (inUsage.get(tgtIndex) ?? 0) + 1);

        return {
            id: `${prefix}_c${idx}`,
            sourceMachineId: sourceMachine.id,
            sourcePortIndex,
            targetMachineId: targetMachine.id,
            targetPortIndex,
        };
    });
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
    const srcLimit = outputPortCountForType(machines[srcIndex].type);
    const tgtLimit = inputPortCountForType(machines[tgtIndex].type);
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
    if (roll < 0.75) return MachineType.M6x4;
    return MachineType.M5x5;
}

function outputPortCountForType(type: MachineType): number {
    switch (type) {
        case MachineType.M3x1:
            return 1;
        case MachineType.M3x3:
            return 3;
        case MachineType.M6x4:
            return 6;
        case MachineType.M5x5:
            return 5;
    }
}

function inputPortCountForType(type: MachineType): number {
    switch (type) {
        case MachineType.M3x1:
            return 0;
        case MachineType.M3x3:
            return 3;
        case MachineType.M6x4:
            return 6;
        case MachineType.M5x5:
            return 5;
    }
}
