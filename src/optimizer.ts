import {
    type GridState,
    type Machine,
    type Connection,
    Orientation,
    getOrientedDimensions,
    getInputPortCount,
    getOutputPortCount,
    isImmovableMachineType,
    normalizeMachineType,
} from './types';
import { createGrid, placeMachine, getMachinePorts, cloneGridState, getPortExternalTile } from './grid';
import { findBeltPath, applyBeltPath, estimateBeltLength } from './pathfinder';
import { evaluateGrid, type ScoreBreakdown, BELT_WEIGHT, AREA_WEIGHT, CORNER_WEIGHT } from './scoring';

const ORIENTATIONS = [Orientation.NORTH, Orientation.EAST, Orientation.SOUTH, Orientation.WEST];

export interface OptimizerResult {
    grid: GridState;
    score: ScoreBreakdown;
    iterations: number;
}

type RandomFn = () => number;
type OptimizerMode = 'normal' | 'deep';

interface SAProfile {
    initialTemp: number;
    coolingRate: number;
    minTemp: number;
    iterPerTemp: number;
    batchSize: number;
    useFastScore: boolean;
}

interface OptimizerRuntimeConfig {
    mode: OptimizerMode;
    timeBudgetMs: number;
    phase1Restarts: number;
    phase2Attempts: number;
    localPolishPasses: number;
    useExplorationSeeds: boolean;
    elitePoolSize: number;
    eliteDiversityHash: boolean;
    eliteMinDistance: number;
    largeMoveRate: number;
    clusterMoveMinSize: number;
    clusterMoveMaxSize: number;
    adaptiveOps: boolean;
    adaptiveWindow: number;
    adaptiveWarmupIterations: number;
    adaptiveMaxOperatorProb: number;
    adaptiveStagnationResetWindow: number;
    adaptiveFlattenFactor: number;
    persistEliteArchive: boolean;
    largeMoveRateEarly: number;
    largeMoveRateLate: number;
    largeMoveCooldownAfterImprove: number;
    criticalNetRate: number;
    repairBeamWidth: number;
    seed?: number;
    phase1SA: SAProfile;
    phase2SA: SAProfile;
}

type MoveOperatorId =
    | 'move_toward_neighbor'
    | 'move_to_source'
    | 'port_facing_jump'
    | 'try_different_port'
    | 'random_shift'
    | 'swap_positions'
    | 'rotate_best'
    | 'joint_move_rotate'
    | 'cluster_destroy_repair'
    | 'critical_net_focus';

interface MoveOperator {
    id: MoveOperatorId;
    baseWeight: number;
    minProbability: number;
}

interface SAOperatorConfig {
    largeMoveRate: number;
    clusterMoveMinSize: number;
    clusterMoveMaxSize: number;
    adaptiveOps: boolean;
    adaptiveWindow: number;
    adaptiveWarmupIterations: number;
    adaptiveMaxOperatorProb: number;
    adaptiveStagnationResetWindow: number;
    adaptiveFlattenFactor: number;
    largeMoveRateEarly: number;
    largeMoveRateLate: number;
    largeMoveCooldownAfterImprove: number;
    criticalNetRate: number;
    repairBeamWidth: number;
}

interface EliteArchiveEntry {
    machines: Machine[];
    connections: Connection[];
    score: number;
    fingerprint: string;
}

interface SerializedEliteArchiveEntry {
    machines: Machine[];
    connections: Connection[];
}

const NORMAL_PHASE1_SA: SAProfile = {
    initialTemp: 100,
    coolingRate: 0.965,
    minTemp: 0.3,
    iterPerTemp: 12,
    batchSize: 50,
    useFastScore: true,
};

const NORMAL_PHASE2_SA: SAProfile = {
    initialTemp: 30,
    coolingRate: 0.98,
    minTemp: 0.1,
    iterPerTemp: 6,
    batchSize: 30,
    useFastScore: false,
};

const DEEP_PHASE1_SA: SAProfile = {
    initialTemp: 130,
    coolingRate: 0.975,
    minTemp: 0.08,
    iterPerTemp: 16,
    batchSize: 72,
    useFastScore: true,
};

const DEEP_PHASE2_SA: SAProfile = {
    initialTemp: 45,
    coolingRate: 0.985,
    minTemp: 0.04,
    iterPerTemp: 10,
    batchSize: 44,
    useFastScore: false,
};

const UNROUTABLE_BASE_PENALTY = 2000;
const UNROUTABLE_PER_CONNECTION_PENALTY = 60;
const UNROUTABLE_PER_MACHINE_PENALTY = 25;

function buildFixedMachineMap(machines: Machine[]): Map<string, Machine> {
    const fixed = new Map<string, Machine>();
    for (const machine of machines) {
        if (!isImmovableMachineType(machine.type)) continue;
        fixed.set(machine.id, { ...machine });
    }
    return fixed;
}

function enforceFixedMachines(machines: Machine[], fixedMachines: Map<string, Machine>): void {
    if (fixedMachines.size === 0) return;
    for (const machine of machines) {
        const fixed = fixedMachines.get(machine.id);
        if (!fixed) continue;
        machine.x = fixed.x;
        machine.y = fixed.y;
        machine.orientation = fixed.orientation;
    }
}

// ─────────────────────────────────────────────────────────
// PUBLIC API
// ─────────────────────────────────────────────────────────

/**
 * Multi-phase optimizer:
 *   Phase 0: Port-aware greedy placement (joint position+orientation search)
 *   Phase 1: Fast SA with Manhattan proxy
 *   Phase 2: A*-verified SA fine-tuning
 *   Phase 3: Port assignment optimization
 *   Phase 4: Compaction + final orientation polish
 */
export async function runOptimizer(
    grid: GridState,
    _config: Record<string, unknown> = {},
    onProgress?: (best: ScoreBreakdown, iteration: number, phase: string) => void,
): Promise<OptimizerResult> {
    const config = normalizeOptimizerConfig(_config);
    const random = createSeededRandom(config.seed);

    const machines = Array.from(grid.machines.values()).map((m) => ({ ...m }));
    const connections = Array.from(grid.connections.values()).map((c) => ({ ...c }));
    const fixedMachines = buildFixedMachineMap(machines);
    const hasFixedMachines = fixedMachines.size > 0;

    if (machines.length === 0) {
        return { grid: cloneGridState(grid), score: evaluateGrid(grid), iterations: 0 };
    }

    const W = grid.width;
    const H = grid.height;
    let totalIter = 0;

    const baselineGrid = cloneGridState(grid);
    const baselineScore = evaluateGrid(grid);
    const budgetStart = performance.now();
    const budgetDeadline = Number.isFinite(config.timeBudgetMs)
        ? budgetStart + config.timeBudgetMs
        : Number.POSITIVE_INFINITY;
    const shouldStopForBudget = (): boolean => config.mode === 'deep' && performance.now() >= budgetDeadline;
    const operatorConfig: SAOperatorConfig = {
        largeMoveRate: config.largeMoveRate,
        clusterMoveMinSize: Math.min(config.clusterMoveMinSize, config.clusterMoveMaxSize),
        clusterMoveMaxSize: Math.max(config.clusterMoveMinSize, config.clusterMoveMaxSize),
        adaptiveOps: config.adaptiveOps,
        adaptiveWindow: config.adaptiveWindow,
        adaptiveWarmupIterations: config.adaptiveWarmupIterations,
        adaptiveMaxOperatorProb: config.adaptiveMaxOperatorProb,
        adaptiveStagnationResetWindow: config.adaptiveStagnationResetWindow,
        adaptiveFlattenFactor: config.adaptiveFlattenFactor,
        largeMoveRateEarly: config.largeMoveRateEarly,
        largeMoveRateLate: config.largeMoveRateLate,
        largeMoveCooldownAfterImprove: config.largeMoveCooldownAfterImprove,
        criticalNetRate: config.criticalNetRate,
        repairBeamWidth: config.repairBeamWidth,
    };
    const eliteArchive: EliteArchiveEntry[] = [];
    const incomingEliteArchive = config.persistEliteArchive
        ? parseIncomingEliteArchive((_config as { incomingEliteArchive?: unknown }).incomingEliteArchive)
        : [];

    let bestFallback: { grid: GridState; score: ScoreBreakdown } = {
        grid: baselineGrid,
        score: baselineScore,
    };
    const rememberFallback = (candidateMachines: Machine[], candidateConnections: Connection[]): void => {
        enforceFixedMachines(candidateMachines, fixedMachines);
        const built = buildGrid(candidateMachines, candidateConnections, W, H);
        if (!built) return;
        const score = evaluateGrid(built);
        if (score.totalScore < bestFallback.score.totalScore) {
            bestFallback = { grid: cloneGridState(built), score };
        }
    };
    const rememberElite = (candidateMachines: Machine[], candidateConnections: Connection[]): void => {
        if (config.elitePoolSize <= 0) return;
        enforceFixedMachines(candidateMachines, fixedMachines);
        const fingerprint = buildLayoutFingerprint(candidateMachines);
        const routed = buildAndScore(candidateMachines, candidateConnections, W, H);
        const routedScore = routed?.score.totalScore;
        const fastScore = scorePlacement(candidateMachines, candidateConnections, W, H, true);
        const score = Number.isFinite(routedScore)
            ? routedScore!
            : Number.isFinite(fastScore)
                ? fastScore + 1000
                : Infinity;
        if (!Number.isFinite(score)) return;

        if (config.eliteDiversityHash || config.eliteMinDistance > 0) {
            for (const entry of eliteArchive) {
                const sameHash = config.eliteDiversityHash && entry.fingerprint === fingerprint;
                const tooClose = config.eliteMinDistance > 0
                    && layoutDistance(candidateMachines, entry.machines) < config.eliteMinDistance;
                if ((sameHash || tooClose) && score >= entry.score) {
                    return;
                }
            }

            for (let i = eliteArchive.length - 1; i >= 0; i--) {
                const entry = eliteArchive[i];
                const sameHash = config.eliteDiversityHash && entry.fingerprint === fingerprint;
                const tooClose = config.eliteMinDistance > 0
                    && layoutDistance(candidateMachines, entry.machines) < config.eliteMinDistance;
                if (sameHash || tooClose) {
                    eliteArchive.splice(i, 1);
                }
            }
        }

        eliteArchive.push({
            machines: candidateMachines.map((m) => ({ ...m })),
            connections: candidateConnections.map((c) => ({ ...c })),
            score,
            fingerprint,
        });
        eliteArchive.sort((a, b) => a.score - b.score);
        if (eliteArchive.length > config.elitePoolSize) {
            eliteArchive.splice(config.elitePoolSize);
        }
    };
    const tryArchiveRestartSeed = (
        fallback: Machine[],
        useFastScore: boolean,
    ): Machine[] => {
        let seedMachines = fallback.map((m) => ({ ...m }));
        if (eliteArchive.length > 0) {
            const idx = Math.min(
                eliteArchive.length - 1,
                Math.floor(Math.pow(random(), 1.6) * eliteArchive.length),
            );
            const picked = eliteArchive[idx];
            seedMachines = picked.machines.map((m) => ({ ...m }));
            connections.length = 0;
            connections.push(...picked.connections.map((c) => ({ ...c })));
        }

        let kicked = seedMachines.map((m) => ({ ...m }));
        const kickMoves = 1 + Math.floor(random() * 2);
        for (let i = 0; i < kickMoves; i++) {
            kicked = perturbSmart(
                kicked,
                connections,
                W,
                H,
                random,
                operatorConfig,
                undefined,
                new Set(fixedMachines.keys()),
            );
        }
        if (scorePlacement(kicked, connections, W, H, useFastScore) < Infinity) {
            return kicked;
        }
        return seedMachines;
    };
    if (config.persistEliteArchive && incomingEliteArchive.length > 0) {
        for (const archived of incomingEliteArchive) {
            rememberElite(archived.machines, archived.connections);
        }
    }
    rememberElite(machines, connections);

    // ── Phase 0: Build deterministic seeds and pick the best ──
    const useExplorationSeeds = config.useExplorationSeeds && !hasFixedMachines;
    const seedCandidates: { name: string; machines: Machine[] }[] = useExplorationSeeds
        ? [
            { name: 'Greedy placement', machines: portAwareGreedyPlace(machines, connections, W, H) },
            { name: 'Layered topology placement', machines: topologyAwareLayeredPlace(machines, connections, W, H) },
            { name: 'Current layout', machines: machines.map((m) => ({ ...m })) },
        ]
        : [{ name: 'Current layout', machines: machines.map((m) => ({ ...m })) }];

    if (useExplorationSeeds) {
        const patternSeed = patternAwarePlace(machines, connections, W, H);
        if (patternSeed) {
            seedCandidates.unshift({ name: 'Pattern-aware placement', machines: patternSeed });
        }
        const twoLayerSeed = twoLayerExhaustivePlace(machines, connections, W, H);
        if (twoLayerSeed) {
            seedCandidates.push({ name: 'Two-layer exhaustive placement', machines: twoLayerSeed });
        }
    }

    let startMachines = seedCandidates[0].machines.map((m) => ({ ...m }));
    let bestSeedFastScore = Infinity;
    let bestSeedRoutedScore = Infinity;
    let hasRoutedSeed = false;
    let bestPhase0Score: ScoreBreakdown | null = null;

    for (const seed of seedCandidates) {
        if (shouldStopForBudget()) break;

        const candidateMachines = seed.machines.map((m) => ({ ...m }));
        const candidateConnections = connections.map((c) => ({ ...c }));
        enforceFixedMachines(candidateMachines, fixedMachines);
        optimizePortAssignments(candidateMachines, candidateConnections, W, H);

        rememberFallback(candidateMachines, candidateConnections);
        rememberElite(candidateMachines, candidateConnections);
        const scored = buildAndScore(candidateMachines, candidateConnections, W, H);
        if (scored && (!bestPhase0Score || scored.score.totalScore < bestPhase0Score.totalScore)) {
            bestPhase0Score = scored.score;
        }
        if (scored && scored.score.totalScore < bestSeedRoutedScore) {
            bestSeedRoutedScore = scored.score.totalScore;
            hasRoutedSeed = true;
            startMachines = candidateMachines.map((m) => ({ ...m }));
            connections.length = 0;
            connections.push(...candidateConnections.map((c) => ({ ...c })));
        }

        const fastScore = scorePlacement(candidateMachines, candidateConnections, W, H, true);
        if (!hasRoutedSeed && fastScore < bestSeedFastScore) {
            bestSeedFastScore = fastScore;
            startMachines = candidateMachines.map((m) => ({ ...m }));
            connections.length = 0;
            connections.push(...candidateConnections.map((c) => ({ ...c })));
        }
    }

    if (onProgress && bestPhase0Score) {
        onProgress(
            bestPhase0Score,
            0,
            config.mode === 'deep' ? 'Phase 0: Current-layout seed' : 'Phase 0: Seed placement',
        );
    }

    // ── Phase 1: Fast SA with optional restarts ───────────
    let phase1Best = startMachines.map((m) => ({ ...m }));
    let phase1BestScore = scorePlacement(phase1Best, connections, W, H, true);
    const phase1Restarts = Math.max(1, config.phase1Restarts);
    let phase1Seed = phase1Best.map((m) => ({ ...m }));

    for (let restart = 0; restart < phase1Restarts; restart++) {
        if (shouldStopForBudget()) break;

        let runIterations = 0;
        const phase1Machines = await runSAAsync(
            phase1Seed,
            connections,
            W,
            H,
            {
                ...config.phase1SA,
                random,
                shouldStop: shouldStopForBudget,
                operators: operatorConfig,
                fixedMachineIds: new Set(fixedMachines.keys()),
            },
            (best, iter) => {
                runIterations = iter;
                if (!onProgress) return;
                const phaseLabel = phase1Restarts > 1
                    ? `Phase 1: Fast SA (${restart + 1}/${phase1Restarts})`
                    : 'Phase 1: Fast SA';
                onProgress(best, totalIter + iter, phaseLabel);
            },
        );

        totalIter += runIterations;
        enforceFixedMachines(phase1Machines, fixedMachines);
        rememberFallback(phase1Machines, connections);
        rememberElite(phase1Machines, connections);

        const candidateScore = scorePlacement(phase1Machines, connections, W, H, true);
        if (candidateScore < phase1BestScore) {
            phase1BestScore = candidateScore;
            phase1Best = phase1Machines.map((m) => ({ ...m }));
        }

        phase1Seed = phase1Best.map((m) => ({ ...m }));
        if (config.mode === 'deep' && restart + 1 < phase1Restarts) {
            phase1Seed = tryArchiveRestartSeed(phase1Best, true);
        }
    }

    // ── Phase 1.5: Early port assignment optimization ──
    const phase1PortOpt = optimizePortAssignments(phase1Best, connections, W, H);
    enforceFixedMachines(phase1PortOpt, fixedMachines);
    rememberFallback(phase1PortOpt, connections);
    rememberElite(phase1PortOpt, connections);

    // ── Phase 2: A*-verified SA fine-tuning (multi-attempt in deep mode) ──
    let phase2Best = phase1PortOpt.map((m) => ({ ...m }));
    let phase2BestScore = scorePlacement(phase2Best, connections, W, H, false);
    const phase2Attempts = Math.max(1, config.phase2Attempts);
    let phase2Seed = phase2Best.map((m) => ({ ...m }));

    for (let attempt = 0; attempt < phase2Attempts; attempt++) {
        if (shouldStopForBudget()) break;

        let runIterations = 0;
        const phase2Machines = await runSAAsync(
            phase2Seed,
            connections,
            W,
            H,
            {
                ...config.phase2SA,
                random,
                shouldStop: shouldStopForBudget,
                operators: operatorConfig,
                fixedMachineIds: new Set(fixedMachines.keys()),
            },
            (best, iter) => {
                runIterations = iter;
                if (!onProgress) return;
                const phaseLabel = phase2Attempts > 1
                    ? `Phase 2: Fine-tune SA (${attempt + 1}/${phase2Attempts})`
                    : 'Phase 2: Fine-tune SA';
                onProgress(best, totalIter + iter, phaseLabel);
            },
        );

        totalIter += runIterations;
        enforceFixedMachines(phase2Machines, fixedMachines);
        rememberFallback(phase2Machines, connections);
        rememberElite(phase2Machines, connections);

        const candidateScore = scorePlacement(phase2Machines, connections, W, H, false);
        if (candidateScore < phase2BestScore) {
            phase2BestScore = candidateScore;
            phase2Best = phase2Machines.map((m) => ({ ...m }));
        }

        phase2Seed = phase2Best.map((m) => ({ ...m }));
        if (config.mode === 'deep' && attempt + 1 < phase2Attempts) {
            phase2Seed = tryArchiveRestartSeed(phase2Best, false);
        }
    }

    // ── Phase 3: Final port assignment optimization ────
    const phase3Machines = optimizePortAssignments(phase2Best, connections, W, H);
    enforceFixedMachines(phase3Machines, fixedMachines);
    rememberFallback(phase3Machines, connections);
    rememberElite(phase3Machines, connections);

    // ── Phase 4: Compaction + orientation polish ──────
    let polished = phase3Machines.map((m) => ({ ...m }));
    const polishPasses = Math.max(1, config.localPolishPasses);
    for (let pass = 0; pass < polishPasses; pass++) {
        if (shouldStopForBudget()) break;

        const compacted = compactLayout(polished, connections, W, H, fixedMachines);
        rememberFallback(compacted, connections);
        rememberElite(compacted, connections);
        polished = optimizeOrientationsUsingAStar(compacted, connections, W, H, fixedMachines);
        rememberFallback(polished, connections);
        rememberElite(polished, connections);

        if (config.mode === 'deep' && pass + 1 < polishPasses) {
            let runIterations = 0;
            const polishedBySA = await runSAAsync(
                polished,
                connections,
                W,
                H,
                {
                    ...config.phase2SA,
                    initialTemp: Math.max(8, config.phase2SA.initialTemp * 0.55),
                    iterPerTemp: Math.max(4, Math.floor(config.phase2SA.iterPerTemp * 0.65)),
                    batchSize: Math.max(14, Math.floor(config.phase2SA.batchSize * 0.65)),
                    random,
                    shouldStop: shouldStopForBudget,
                    operators: operatorConfig,
                    fixedMachineIds: new Set(fixedMachines.keys()),
                },
                (best, iter) => {
                    runIterations = iter;
                    if (!onProgress) return;
                    onProgress(
                        best,
                        totalIter + iter,
                        `Phase 4: Local polish SA (${pass + 1}/${polishPasses})`,
                    );
                },
            );
            totalIter += runIterations;
            enforceFixedMachines(polishedBySA, fixedMachines);
            rememberFallback(polishedBySA, connections);
            rememberElite(polishedBySA, connections);
            polished = polishedBySA;
        }

        if (onProgress) {
            const phase4Score = buildAndScore(polished, connections, W, H);
            if (phase4Score) {
                const phaseLabel = polishPasses > 1
                    ? `Phase 4: Compaction/orient (${pass + 1}/${polishPasses})`
                    : 'Phase 4: Compaction/orient';
                onProgress(phase4Score.score, totalIter, phaseLabel);
            }
        }
    }

    enforceFixedMachines(polished, fixedMachines);
    rememberElite(polished, connections);
    const finalResult = buildGrid(polished, connections, W, H);
    if (finalResult) {
        const finalScore = evaluateGrid(finalResult);
        if (finalScore.totalScore < bestFallback.score.totalScore) {
            bestFallback = { grid: finalResult, score: finalScore };
        }
    }

    // Never return a worse result than the starting layout.
    if (baselineScore.totalScore < bestFallback.score.totalScore) {
        bestFallback = { grid: baselineGrid, score: baselineScore };
    }

    if (config.persistEliteArchive) {
        (_config as { outgoingEliteArchive?: SerializedEliteArchiveEntry[] }).outgoingEliteArchive
            = serializeEliteArchive(eliteArchive);
    }

    if (onProgress) {
        const doneLabel = shouldStopForBudget() ? 'Done (time budget reached)' : 'Done';
        onProgress(bestFallback.score, totalIter, doneLabel);
    }

    return {
        grid: bestFallback.grid,
        score: bestFallback.score,
        iterations: totalIter,
    };
}

function normalizeOptimizerConfig(config: Record<string, unknown>): OptimizerRuntimeConfig {
    const mode: OptimizerMode = config.mode === 'deep' ? 'deep' : 'normal';
    const defaults = mode === 'deep'
        ? {
            timeBudgetMs: 7000,
            phase1Restarts: 3,
            phase2Attempts: 3,
            localPolishPasses: 3,
            useExplorationSeeds: false,
            elitePoolSize: 12,
            eliteDiversityHash: true,
            eliteMinDistance: 0,
            largeMoveRate: 0.2,
            clusterMoveMinSize: 2,
            clusterMoveMaxSize: 5,
            adaptiveOps: true,
            adaptiveWindow: 120,
            adaptiveWarmupIterations: 0,
            adaptiveMaxOperatorProb: 1,
            adaptiveStagnationResetWindow: Number.MAX_SAFE_INTEGER,
            adaptiveFlattenFactor: 0,
            persistEliteArchive: false,
            largeMoveRateEarly: 0.2,
            largeMoveRateLate: 0.2,
            largeMoveCooldownAfterImprove: 0,
            criticalNetRate: 0,
            repairBeamWidth: 1,
            phase1SA: DEEP_PHASE1_SA,
            phase2SA: DEEP_PHASE2_SA,
        }
        : {
            timeBudgetMs: Number.POSITIVE_INFINITY,
            phase1Restarts: 1,
            phase2Attempts: 1,
            localPolishPasses: 1,
            useExplorationSeeds: true,
            elitePoolSize: 8,
            eliteDiversityHash: true,
            eliteMinDistance: 0,
            largeMoveRate: 0.12,
            clusterMoveMinSize: 2,
            clusterMoveMaxSize: 4,
            adaptiveOps: false,
            adaptiveWindow: 100,
            adaptiveWarmupIterations: 0,
            adaptiveMaxOperatorProb: 1,
            adaptiveStagnationResetWindow: Number.MAX_SAFE_INTEGER,
            adaptiveFlattenFactor: 0,
            persistEliteArchive: false,
            largeMoveRateEarly: 0.12,
            largeMoveRateLate: 0.12,
            largeMoveCooldownAfterImprove: 0,
            criticalNetRate: 0,
            repairBeamWidth: 1,
            phase1SA: NORMAL_PHASE1_SA,
            phase2SA: NORMAL_PHASE2_SA,
        };

    const largeMoveRate = pickClampedNumber(config.largeMoveRate, defaults.largeMoveRate, 0, 0.6);
    const largeMoveRateEarly = pickClampedNumber(config.largeMoveRateEarly, largeMoveRate, 0, 0.7);
    const largeMoveRateLate = pickClampedNumber(config.largeMoveRateLate, largeMoveRate, 0, 0.7);
    const criticalNetRate = pickClampedNumber(config.criticalNetRate, defaults.criticalNetRate, 0, largeMoveRate);

    return {
        mode,
        timeBudgetMs: pickPositiveNumber(config.timeBudgetMs, defaults.timeBudgetMs),
        phase1Restarts: pickPositiveInteger(config.phase1Restarts, defaults.phase1Restarts),
        phase2Attempts: pickPositiveInteger(config.phase2Attempts, defaults.phase2Attempts),
        localPolishPasses: pickPositiveInteger(config.localPolishPasses, defaults.localPolishPasses),
        useExplorationSeeds: pickBoolean(config.useExplorationSeeds, defaults.useExplorationSeeds),
        elitePoolSize: pickPositiveInteger(config.elitePoolSize, defaults.elitePoolSize),
        eliteDiversityHash: pickBoolean(config.eliteDiversityHash, defaults.eliteDiversityHash),
        eliteMinDistance: pickNonNegativeNumber(config.eliteMinDistance, defaults.eliteMinDistance),
        largeMoveRate,
        clusterMoveMinSize: pickPositiveInteger(config.clusterMoveMinSize, defaults.clusterMoveMinSize),
        clusterMoveMaxSize: pickPositiveInteger(config.clusterMoveMaxSize, defaults.clusterMoveMaxSize),
        adaptiveOps: pickBoolean(config.adaptiveOps, defaults.adaptiveOps),
        adaptiveWindow: pickPositiveInteger(config.adaptiveWindow, defaults.adaptiveWindow),
        adaptiveWarmupIterations: pickNonNegativeInteger(
            config.adaptiveWarmupIterations,
            defaults.adaptiveWarmupIterations,
        ),
        adaptiveMaxOperatorProb: pickClampedNumber(
            config.adaptiveMaxOperatorProb,
            defaults.adaptiveMaxOperatorProb,
            0.12,
            1,
        ),
        adaptiveStagnationResetWindow: pickNonNegativeInteger(
            config.adaptiveStagnationResetWindow,
            defaults.adaptiveStagnationResetWindow,
        ),
        adaptiveFlattenFactor: pickClampedNumber(config.adaptiveFlattenFactor, defaults.adaptiveFlattenFactor, 0, 1),
        persistEliteArchive: pickBoolean(config.persistEliteArchive, defaults.persistEliteArchive),
        largeMoveRateEarly,
        largeMoveRateLate,
        largeMoveCooldownAfterImprove: pickNonNegativeInteger(
            config.largeMoveCooldownAfterImprove,
            defaults.largeMoveCooldownAfterImprove,
        ),
        criticalNetRate,
        repairBeamWidth: pickPositiveInteger(config.repairBeamWidth, defaults.repairBeamWidth),
        seed: pickSeed(config.seed),
        phase1SA: defaults.phase1SA,
        phase2SA: defaults.phase2SA,
    };
}

function pickPositiveInteger(value: unknown, fallback: number): number {
    if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
    return Math.max(1, Math.floor(value));
}

function pickNonNegativeInteger(value: unknown, fallback: number): number {
    if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
    return Math.max(0, Math.floor(value));
}

function pickPositiveNumber(value: unknown, fallback: number): number {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return fallback;
    return value;
}

function pickNonNegativeNumber(value: unknown, fallback: number): number {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return fallback;
    return value;
}

function pickClampedNumber(value: unknown, fallback: number, min: number, max: number): number {
    if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
    return Math.max(min, Math.min(max, value));
}

function pickBoolean(value: unknown, fallback: boolean): boolean {
    if (typeof value !== 'boolean') return fallback;
    return value;
}

function pickSeed(value: unknown): number | undefined {
    if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
    return Math.floor(value);
}

function createSeededRandom(seed?: number): RandomFn {
    if (seed === undefined) return Math.random;
    let state = (seed >>> 0) || 0x6d2b79f5;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 0x100000000;
    };
}

function parseIncomingEliteArchive(value: unknown): SerializedEliteArchiveEntry[] {
    if (!Array.isArray(value)) return [];
    const parsed: SerializedEliteArchiveEntry[] = [];

    for (const rawEntry of value) {
        if (!rawEntry || typeof rawEntry !== 'object') continue;
        const machinesRaw = (rawEntry as { machines?: unknown }).machines;
        const connectionsRaw = (rawEntry as { connections?: unknown }).connections;
        if (!Array.isArray(machinesRaw) || !Array.isArray(connectionsRaw)) continue;

        const machines: Machine[] = [];
        for (const rawMachine of machinesRaw) {
            if (!rawMachine || typeof rawMachine !== 'object') continue;
            const machine = rawMachine as Partial<Machine>;
            const type = normalizeMachineType(machine.type);
            if (
                typeof machine.id !== 'string'
                || !type
                || typeof machine.x !== 'number'
                || !Number.isFinite(machine.x)
                || typeof machine.y !== 'number'
                || !Number.isFinite(machine.y)
                || typeof machine.orientation !== 'string'
            ) {
                continue;
            }
            machines.push({
                id: machine.id,
                type,
                x: machine.x,
                y: machine.y,
                orientation: machine.orientation as Orientation,
            });
        }

        const connections: Connection[] = [];
        for (const rawConnection of connectionsRaw) {
            if (!rawConnection || typeof rawConnection !== 'object') continue;
            const connection = rawConnection as Partial<Connection>;
            if (
                typeof connection.id !== 'string'
                || typeof connection.sourceMachineId !== 'string'
                || typeof connection.targetMachineId !== 'string'
                || typeof connection.sourcePortIndex !== 'number'
                || !Number.isFinite(connection.sourcePortIndex)
                || typeof connection.targetPortIndex !== 'number'
                || !Number.isFinite(connection.targetPortIndex)
            ) {
                continue;
            }
            connections.push({
                id: connection.id,
                sourceMachineId: connection.sourceMachineId,
                sourcePortIndex: Math.floor(connection.sourcePortIndex),
                targetMachineId: connection.targetMachineId,
                targetPortIndex: Math.floor(connection.targetPortIndex),
            });
        }

        if (machines.length === 0 || connections.length === 0) continue;
        parsed.push({ machines, connections });
    }

    return parsed;
}

function serializeEliteArchive(entries: EliteArchiveEntry[]): SerializedEliteArchiveEntry[] {
    return entries.map((entry) => ({
        machines: entry.machines.map((machine) => ({ ...machine })),
        connections: entry.connections.map((connection) => ({ ...connection })),
    }));
}

// ─────────────────────────────────────────────────────────
// PHASE 0: PORT-AWARE GREEDY PLACEMENT
// ─────────────────────────────────────────────────────────

/**
 * Port-aware greedy placement:
 * For each machine to place, test ALL 4 sides of its best neighbor
 * × ALL 4 orientations of the machine being placed, picking the
 * (position, orientation) combo that minimizes total Manhattan port distance.
 */
function portAwareGreedyPlace(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
): Machine[] {
    if (machines.length === 0) return [];

    // Build adjacency counts
    const adjCount = new Map<string, Map<string, number>>();
    for (const m of machines) adjCount.set(m.id, new Map());
    for (const c of connections) {
        const src = adjCount.get(c.sourceMachineId);
        const tgt = adjCount.get(c.targetMachineId);
        if (!src || !tgt) continue;
        src.set(c.targetMachineId, (src.get(c.targetMachineId) || 0) + 1);
        tgt.set(c.sourceMachineId, (tgt.get(c.sourceMachineId) || 0) + 1);
    }

    // Determine placement order: start with most-connected machine
    const totalConnections = new Map<string, number>();
    for (const [id, neighbors] of adjCount) {
        let total = 0;
        for (const count of neighbors.values()) total += count;
        totalConnections.set(id, total);
    }

    const result = machines.map((m) => ({ ...m }));
    const placed = new Set<string>();
    const machineMap = new Map(result.map((m) => [m.id, m]));

    // Find the most-connected pair for initial placement
    let bestPair: [string, string] | null = null;
    let bestPairCount = 0;
    for (const [id, neighbors] of adjCount) {
        for (const [nid, count] of neighbors) {
            if (count > bestPairCount) {
                bestPairCount = count;
                bestPair = [id, nid];
            }
        }
    }

    // ── Place first machine ──────────────────────────────
    const firstId = bestPair ? bestPair[0] : machines[0].id;
    const first = machineMap.get(firstId)!;

    // Try all orientations for first machine and pick smallest footprint
    let bestFirstOrient = first.orientation;
    let smallestArea = Infinity;
    for (const orient of ORIENTATIONS) {
        first.orientation = orient;
        const dims = getOrientedDimensions(first);
        const area = dims.width * dims.height;
        if (area < smallestArea) {
            smallestArea = area;
            bestFirstOrient = orient;
        }
    }
    first.orientation = bestFirstOrient;

    // Place first machine at a good starting position (enough room for belt gaps)
    first.x = 2;
    first.y = 2;
    placed.add(firstId);

    // ── Place second machine of the best pair using port-facing ──
    if (bestPair) {
        const secondId = bestPair[1];
        const second = machineMap.get(secondId)!;
        const bestPlacement = findBestPortFacingPosition(
            second, first, connections, result, placed, gridW, gridH, machineMap,
        );
        if (bestPlacement) {
            second.x = bestPlacement.x;
            second.y = bestPlacement.y;
            second.orientation = bestPlacement.orientation;
        }
        placed.add(secondId);
    }

    // ── Place remaining machines ─────────────────────────
    while (placed.size < result.length) {
        // Pick unplaced machine with most connections to placed machines
        let bestUnplaced: string | null = null;
        let bestConnectivity = -1;

        for (const m of result) {
            if (placed.has(m.id)) continue;
            const neighbors = adjCount.get(m.id)!;
            let connectivity = 0;
            for (const [nid, count] of neighbors) {
                if (placed.has(nid)) connectivity += count;
            }
            if (connectivity > bestConnectivity || (connectivity === bestConnectivity && bestUnplaced === null)) {
                bestConnectivity = connectivity;
                bestUnplaced = m.id;
            }
        }

        if (!bestUnplaced) {
            bestUnplaced = result.find((m) => !placed.has(m.id))!.id;
        }

        const machine = machineMap.get(bestUnplaced)!;

        // Find ALL placed neighbors, sorted by connection count (most first)
        const neighbors = adjCount.get(bestUnplaced)!;
        const placedNeighbors: { id: string; count: number }[] = [];
        for (const [nid, count] of neighbors) {
            if (placed.has(nid)) {
                placedNeighbors.push({ id: nid, count });
            }
        }
        placedNeighbors.sort((a, b) => b.count - a.count);
        const bestNeighborId = placedNeighbors.length > 0 ? placedNeighbors[0].id : null;

        if (bestNeighborId) {
            // Try placement relative to EACH placed neighbor, pick globally best
            let globalBest: { x: number; y: number; orientation: Orientation; cost: number } | null = null;
            for (const pn of placedNeighbors) {
                const neighbor = machineMap.get(pn.id)!;
                const placement = findBestPortFacingPosition(
                    machine, neighbor, connections, result, placed, gridW, gridH, machineMap,
                );
                if (!placement) continue;
                // Score this placement against ALL placed neighbors
                machine.x = placement.x;
                machine.y = placement.y;
                machine.orientation = placement.orientation;
                let totalCost = 0;
                for (const conn of connections) {
                    if (conn.sourceMachineId !== machine.id && conn.targetMachineId !== machine.id) continue;
                    const otherId = conn.sourceMachineId === machine.id ? conn.targetMachineId : conn.sourceMachineId;
                    if (!placed.has(otherId)) continue;
                    const src = machineMap.get(conn.sourceMachineId);
                    const tgt = machineMap.get(conn.targetMachineId);
                    if (!src || !tgt) continue;
                    const srcPorts = getMachinePorts(src);
                    const tgtPorts = getMachinePorts(tgt);
                    const srcPort = srcPorts.outputs[conn.sourcePortIndex];
                    const tgtPort = tgtPorts.inputs[conn.targetPortIndex];
                    if (srcPort && tgtPort) totalCost += estimateBeltLength(srcPort, tgtPort);
                }
                if (!globalBest || totalCost < globalBest.cost) {
                    globalBest = { x: placement.x, y: placement.y, orientation: placement.orientation, cost: totalCost };
                }
            }
            const bestPlacement = globalBest;
            if (bestPlacement) {
                machine.x = bestPlacement.x;
                machine.y = bestPlacement.y;
                machine.orientation = bestPlacement.orientation;
            } else {
                // Fallback: spiral search with best orientation
                const neighbor = machineMap.get(bestNeighborId)!;
                placeWithSpiralAndOrient(machine, neighbor, connections, result, placed, gridW, gridH, machineMap);
            }
        } else {
            // No connected neighbor — spiral from first placed machine
            const firstPlaced = result.find((m) => placed.has(m.id))!;
            placeWithSpiralAndOrient(machine, firstPlaced, connections, result, placed, gridW, gridH, machineMap);
        }

        placed.add(bestUnplaced);
    }

    return result;
}

/**
 * Find the best port-facing position for `machine` relative to `neighbor`.
 * Tests all 4 sides × all 4 orientations and picks the combo with the
 * lowest total Manhattan port distance across all connections involving this machine.
 */
function findBestPortFacingPosition(
    machine: Machine,
    neighbor: Machine,
    connections: Connection[],
    allMachines: Machine[],
    placed: Set<string>,
    gridW: number,
    gridH: number,
    machineMap: Map<string, Machine>,
): { x: number; y: number; orientation: Orientation } | null {
    const nDims = getOrientedDimensions(neighbor);
    let bestResult: { x: number; y: number; orientation: Orientation } | null = null;
    let bestCost = Infinity;

    for (const orient of ORIENTATIONS) {
        machine.orientation = orient;
        const dims = getOrientedDimensions(machine);

        // Generate candidate positions: 4 sides of neighbor with 1-tile belt gap
        // Also try aligned positions (centered, left-aligned, right-aligned)
        const sidePositions: { x: number; y: number }[] = [];

        // Below neighbor
        for (let dx = -(dims.width - 1); dx <= nDims.width - 1; dx++) {
            sidePositions.push({ x: neighbor.x + dx, y: neighbor.y + nDims.height + 1 });
        }
        // Above neighbor
        for (let dx = -(dims.width - 1); dx <= nDims.width - 1; dx++) {
            sidePositions.push({ x: neighbor.x + dx, y: neighbor.y - dims.height - 1 });
        }
        // Right of neighbor
        for (let dy = -(dims.height - 1); dy <= nDims.height - 1; dy++) {
            sidePositions.push({ x: neighbor.x + nDims.width + 1, y: neighbor.y + dy });
        }
        // Left of neighbor
        for (let dy = -(dims.height - 1); dy <= nDims.height - 1; dy++) {
            sidePositions.push({ x: neighbor.x - dims.width - 1, y: neighbor.y + dy });
        }

        for (const pos of sidePositions) {
            machine.x = pos.x;
            machine.y = pos.y;

            if (!isValidPlacement(machine, allMachines, placed, gridW, gridH)) continue;

            // Score: total Manhattan distance for all connections involving this machine
            let totalDist = 0;
            let validConns = true;
            for (const conn of connections) {
                if (conn.sourceMachineId !== machine.id && conn.targetMachineId !== machine.id) continue;
                const src = machineMap.get(conn.sourceMachineId);
                const tgt = machineMap.get(conn.targetMachineId);
                if (!src || !tgt) continue;
                // Only score connections where the other end is already placed
                const otherId = conn.sourceMachineId === machine.id ? conn.targetMachineId : conn.sourceMachineId;
                if (!placed.has(otherId) && otherId !== neighbor.id) continue;

                const srcPorts = getMachinePorts(src);
                const tgtPorts = getMachinePorts(tgt);
                const srcPort = srcPorts.outputs[conn.sourcePortIndex];
                const tgtPort = tgtPorts.inputs[conn.targetPortIndex];
                if (!srcPort || !tgtPort) { validConns = false; break; }
                totalDist += estimateBeltLength(srcPort, tgtPort);
            }
            if (!validConns) continue;

            if (totalDist < bestCost) {
                bestCost = totalDist;
                bestResult = { x: pos.x, y: pos.y, orientation: orient };
            }
        }
    }

    return bestResult;
}

function placeWithSpiralAndOrient(
    machine: Machine,
    anchor: Machine,
    connections: Connection[],
    allMachines: Machine[],
    placed: Set<string>,
    gridW: number,
    gridH: number,
    machineMap: Map<string, Machine>,
): void {
    let bestCost = Infinity;
    let bestX = anchor.x;
    let bestY = anchor.y + 5;
    let bestOrient = machine.orientation;

    for (const orient of ORIENTATIONS) {
        machine.orientation = orient;
        const spiralPos = spiralSearch(anchor.x, anchor.y, machine, allMachines, placed, gridW, gridH);
        if (!spiralPos) continue;

        machine.x = spiralPos.x;
        machine.y = spiralPos.y;

        let totalDist = 0;
        for (const conn of connections) {
            if (conn.sourceMachineId !== machine.id && conn.targetMachineId !== machine.id) continue;
            const src = machineMap.get(conn.sourceMachineId);
            const tgt = machineMap.get(conn.targetMachineId);
            if (!src || !tgt) continue;
            const otherId = conn.sourceMachineId === machine.id ? conn.targetMachineId : conn.sourceMachineId;
            if (!placed.has(otherId)) continue;

            const srcPorts = getMachinePorts(src);
            const tgtPorts = getMachinePorts(tgt);
            const srcPort = srcPorts.outputs[conn.sourcePortIndex];
            const tgtPort = tgtPorts.inputs[conn.targetPortIndex];
            if (srcPort && tgtPort) totalDist += estimateBeltLength(srcPort, tgtPort);
        }

        if (totalDist < bestCost) {
            bestCost = totalDist;
            bestX = spiralPos.x;
            bestY = spiralPos.y;
            bestOrient = orient;
        }
    }

    machine.x = bestX;
    machine.y = bestY;
    machine.orientation = bestOrient;
}

/**
 * Topology-aware deterministic placement:
 * - Layer machines by graph depth (longest-path from sources)
 * - Order each layer with barycentric sweeps to reduce crossings
 * - Place layers compactly with guaranteed non-overlap
 */
function topologyAwareLayeredPlace(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
): Machine[] {
    if (machines.length === 0) return [];

    const result = machines.map((m) => ({ ...m, orientation: Orientation.NORTH }));

    const incoming = new Map<string, string[]>();
    const outgoing = new Map<string, string[]>();
    const indegree = new Map<string, number>();

    for (const machine of result) {
        incoming.set(machine.id, []);
        outgoing.set(machine.id, []);
        indegree.set(machine.id, 0);
    }

    for (const conn of connections) {
        if (!incoming.has(conn.targetMachineId) || !outgoing.has(conn.sourceMachineId)) continue;
        incoming.get(conn.targetMachineId)!.push(conn.sourceMachineId);
        outgoing.get(conn.sourceMachineId)!.push(conn.targetMachineId);
        indegree.set(conn.targetMachineId, (indegree.get(conn.targetMachineId) || 0) + 1);
    }

    const totalDegree = new Map<string, number>();
    for (const machine of result) {
        totalDegree.set(
            machine.id,
            (incoming.get(machine.id)?.length || 0) + (outgoing.get(machine.id)?.length || 0),
        );
    }

    const queue = result
        .map((m) => m.id)
        .filter((id) => (indegree.get(id) || 0) === 0)
        .sort((a, b) => {
            const degreeDelta = (totalDegree.get(b) || 0) - (totalDegree.get(a) || 0);
            if (degreeDelta !== 0) return degreeDelta;
            return a.localeCompare(b);
        });

    const topoOrder: string[] = [];
    while (queue.length > 0) {
        const id = queue.shift()!;
        topoOrder.push(id);
        for (const nextId of outgoing.get(id) || []) {
            indegree.set(nextId, (indegree.get(nextId) || 0) - 1);
            if ((indegree.get(nextId) || 0) === 0) queue.push(nextId);
        }
        queue.sort((a, b) => {
            const degreeDelta = (totalDegree.get(b) || 0) - (totalDegree.get(a) || 0);
            if (degreeDelta !== 0) return degreeDelta;
            return a.localeCompare(b);
        });
    }

    if (topoOrder.length < result.length) {
        const remaining = result
            .map((m) => m.id)
            .filter((id) => !topoOrder.includes(id))
            .sort((a, b) => {
                const degreeDelta = (totalDegree.get(b) || 0) - (totalDegree.get(a) || 0);
                if (degreeDelta !== 0) return degreeDelta;
                return a.localeCompare(b);
            });
        topoOrder.push(...remaining);
    }

    const topoIndex = new Map<string, number>();
    for (let i = 0; i < topoOrder.length; i++) {
        topoIndex.set(topoOrder[i], i);
    }

    const layerById = new Map<string, number>();
    for (const id of topoOrder) {
        let layer = 0;
        for (const pred of incoming.get(id) || []) {
            layer = Math.max(layer, (layerById.get(pred) || 0) + 1);
        }
        layerById.set(id, layer);
    }

    let maxLayer = 0;
    for (const layer of layerById.values()) {
        if (layer > maxLayer) maxLayer = layer;
    }

    const byId = new Map(result.map((m) => [m.id, m]));
    const layers: Machine[][] = Array.from({ length: maxLayer + 1 }, () => []);
    for (const [id, layer] of layerById) {
        const machine = byId.get(id);
        if (machine) layers[layer].push(machine);
    }

    for (const layerMachines of layers) {
        layerMachines.sort((a, b) => (topoIndex.get(a.id) || 0) - (topoIndex.get(b.id) || 0));
    }

    const neighborBarycenter = (
        ids: string[],
        indexMap: Map<string, number>,
        fallback: number,
    ): number => {
        if (ids.length === 0) return fallback;
        let total = 0;
        let count = 0;
        for (const id of ids) {
            const idx = indexMap.get(id);
            if (idx === undefined) continue;
            total += idx;
            count++;
        }
        return count > 0 ? total / count : fallback;
    };

    for (let pass = 0; pass < 2; pass++) {
        for (let layer = 1; layer <= maxLayer; layer++) {
            const prevOrder = new Map<string, number>();
            layers[layer - 1].forEach((m, idx) => prevOrder.set(m.id, idx));
            layers[layer].sort((a, b) => {
                const aBary = neighborBarycenter(incoming.get(a.id) || [], prevOrder, topoIndex.get(a.id) || 0);
                const bBary = neighborBarycenter(incoming.get(b.id) || [], prevOrder, topoIndex.get(b.id) || 0);
                return aBary - bBary;
            });
        }

        for (let layer = maxLayer - 1; layer >= 0; layer--) {
            const nextOrder = new Map<string, number>();
            layers[layer + 1].forEach((m, idx) => nextOrder.set(m.id, idx));
            layers[layer].sort((a, b) => {
                const aBary = neighborBarycenter(outgoing.get(a.id) || [], nextOrder, topoIndex.get(a.id) || 0);
                const bBary = neighborBarycenter(outgoing.get(b.id) || [], nextOrder, topoIndex.get(b.id) || 0);
                return aBary - bBary;
            });
        }
    }

    const marginX = 1;
    const marginY = 1;
    const defaultGapX = 2;
    const defaultGapY = 3;

    const layerHeights = layers.map((layerMachines) => {
        let height = 0;
        for (const machine of layerMachines) {
            height = Math.max(height, getOrientedDimensions(machine).height);
        }
        return height;
    });

    const totalHeight = layerHeights.reduce((sum, h) => sum + h, 0);
    const layerCount = layers.length;
    const availableGapY = gridH - (marginY * 2) - totalHeight;
    const gapY =
        layerCount > 1
            ? availableGapY >= (layerCount - 1)
                ? Math.min(defaultGapY, Math.floor(availableGapY / (layerCount - 1)))
                : 0
            : 0;

    let yCursor = marginY;
    for (let layer = 0; layer < layers.length; layer++) {
        const layerMachines = layers[layer];
        if (layerMachines.length === 0) continue;

        const machineWidths = layerMachines.reduce((sum, m) => sum + getOrientedDimensions(m).width, 0);
        const slots = layerMachines.length - 1;
        const maxGapX = slots > 0 ? Math.floor((gridW - (marginX * 2) - machineWidths) / slots) : 0;
        const gapX = slots > 0 ? (maxGapX >= 1 ? Math.min(defaultGapX, maxGapX) : 0) : 0;
        const layerWidth = machineWidths + (gapX * slots);

        let xCursor = Math.max(0, Math.floor((gridW - layerWidth) / 2));
        for (const machine of layerMachines) {
            const dims = getOrientedDimensions(machine);
            machine.x = xCursor;
            machine.y = Math.max(0, Math.min(gridH - dims.height, yCursor));
            xCursor += dims.width + gapX;
        }

        yCursor += layerHeights[layer] + gapY;
    }

    // If any placement is invalid after clamping/packing, repair via spiral relocation.
    const placed = new Set<string>();
    const sorted = result.slice().sort((a, b) => (a.y - b.y) || (a.x - b.x));
    for (const machine of sorted) {
        if (!isValidPlacement(machine, result, placed, gridW, gridH)) {
            const anchorX = Math.max(0, Math.min(gridW - 1, machine.x));
            const anchorY = Math.max(0, Math.min(gridH - 1, machine.y));
            const repaired = spiralSearch(anchorX, anchorY, machine, result, placed, gridW, gridH);
            if (repaired) {
                machine.x = repaired.x;
                machine.y = repaired.y;
            }
        }
        placed.add(machine.id);
    }

    return result;
}

/**
 * Exact two-layer seed:
 * when the graph is strictly layer-0 -> layer-1, brute-force row ordering
 * and keep the best A*-validated arrangement.
 */
function twoLayerExhaustivePlace(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
): Machine[] | null {
    if (machines.length < 2) return null;

    const base = machines.map((m) => ({ ...m, orientation: Orientation.NORTH }));
    const incoming = new Map<string, string[]>();
    const outgoing = new Map<string, string[]>();
    const indegree = new Map<string, number>();

    for (const machine of base) {
        incoming.set(machine.id, []);
        outgoing.set(machine.id, []);
        indegree.set(machine.id, 0);
    }
    for (const conn of connections) {
        if (!incoming.has(conn.targetMachineId) || !outgoing.has(conn.sourceMachineId)) continue;
        incoming.get(conn.targetMachineId)!.push(conn.sourceMachineId);
        outgoing.get(conn.sourceMachineId)!.push(conn.targetMachineId);
        indegree.set(conn.targetMachineId, (indegree.get(conn.targetMachineId) || 0) + 1);
    }

    const queue = base.map((m) => m.id).filter((id) => (indegree.get(id) || 0) === 0);
    const topoOrder: string[] = [];
    while (queue.length > 0) {
        const id = queue.shift()!;
        topoOrder.push(id);
        for (const nextId of outgoing.get(id) || []) {
            indegree.set(nextId, (indegree.get(nextId) || 0) - 1);
            if ((indegree.get(nextId) || 0) === 0) queue.push(nextId);
        }
    }
    if (topoOrder.length !== base.length) return null;

    const layerById = new Map<string, number>();
    for (const id of topoOrder) {
        let layer = 0;
        for (const pred of incoming.get(id) || []) {
            layer = Math.max(layer, (layerById.get(pred) || 0) + 1);
        }
        layerById.set(id, layer);
    }

    let maxLayer = 0;
    for (const layer of layerById.values()) {
        if (layer > maxLayer) maxLayer = layer;
    }
    if (maxLayer !== 1) return null;

    const byId = new Map(base.map((m) => [m.id, m]));
    const topLayer: Machine[] = [];
    const bottomLayer: Machine[] = [];
    for (const [id, layer] of layerById) {
        const machine = byId.get(id);
        if (!machine) continue;
        if (layer === 0) topLayer.push(machine);
        else if (layer === 1) bottomLayer.push(machine);
    }
    if (topLayer.length === 0 || bottomLayer.length === 0) return null;

    const permutationBudget = factorial(topLayer.length) * factorial(bottomLayer.length);
    if (permutationBudget > 4000) return null;

    const topPerms = generatePermutations(topLayer);
    const bottomPerms = generatePermutations(bottomLayer);

    let best: { machines: Machine[]; score: number } | null = null;
    for (const topOrder of topPerms) {
        for (const bottomOrder of bottomPerms) {
            const candidate = base.map((m) => ({ ...m }));
            const candidateById = new Map(candidate.map((m) => [m.id, m]));
            const top = topOrder.map((m) => candidateById.get(m.id)!);
            const bottom = bottomOrder.map((m) => candidateById.get(m.id)!);

            const topHeight = top.reduce((h, m) => Math.max(h, getOrientedDimensions(m).height), 0);
            const yTop = 1;
            const yBottom = yTop + topHeight + 2;

            if (!placeRowCentered(top, yTop, gridW)) continue;
            if (!placeRowCentered(bottom, yBottom, gridW)) continue;

            const allIds = new Set(candidate.map((m) => m.id));
            let valid = true;
            for (const machine of candidate) {
                if (!isValidPlacement(machine, candidate, allIds, gridW, gridH)) {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;

            const candidateConnections = connections.map((c) => ({ ...c }));
            optimizePortAssignments(candidate, candidateConnections, gridW, gridH);
            const scored = buildAndScore(candidate, candidateConnections, gridW, gridH);
            if (!scored) continue;

            if (!best || scored.score.totalScore < best.score) {
                best = {
                    machines: candidate.map((m) => ({ ...m })),
                    score: scored.score.totalScore,
                };
            }
        }
    }

    return best ? best.machines : null;
}

function placeRowCentered(row: Machine[], y: number, gridW: number): boolean {
    return placeRowCenteredWithGap(row, y, gridW, 2);
}

function placeRowCenteredWithGap(
    row: Machine[],
    y: number,
    gridW: number,
    preferredGap: number,
): boolean {
    if (row.length === 0) return true;

    const widths = row.map((m) => getOrientedDimensions(m).width);
    const widthTotal = widths.reduce((sum, w) => sum + w, 0);
    let gap = row.length > 1 ? preferredGap : 0;

    while (gap > 0 && (widthTotal + gap * (row.length - 1)) > gridW) {
        gap--;
    }

    const rowWidth = widthTotal + gap * (row.length - 1);
    if (rowWidth > gridW) return false;

    let x = Math.floor((gridW - rowWidth) / 2);
    for (let i = 0; i < row.length; i++) {
        row[i].x = x;
        row[i].y = y;
        x += widths[i] + gap;
    }
    return true;
}

function factorial(n: number): number {
    let result = 1;
    for (let i = 2; i <= n; i++) result *= i;
    return result;
}

function generatePermutations<T>(items: T[]): T[][] {
    const results: T[][] = [];
    const used = new Array(items.length).fill(false);
    const current: T[] = [];

    function dfs(): void {
        if (current.length === items.length) {
            results.push([...current]);
            return;
        }

        for (let i = 0; i < items.length; i++) {
            if (used[i]) continue;
            used[i] = true;
            current.push(items[i]);
            dfs();
            current.pop();
            used[i] = false;
        }
    }

    dfs();
    return results;
}

function patternAwarePlace(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
): Machine[] | null {
    const layered = layeredFlowSeed(machines, connections, gridW, gridH);
    if (layered) return layered;

    const ring = ringChordSeed(machines, connections, gridW, gridH);
    if (ring) return ring;

    return null;
}

function layeredFlowSeed(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
): Machine[] | null {
    if (machines.length < 6) return null;

    const result = machines.map((m) => ({ ...m, orientation: Orientation.NORTH }));
    const initialOrder = new Map(machines.map((m, idx) => [m.id, idx]));
    const incoming = new Map<string, string[]>();
    const outgoing = new Map<string, string[]>();

    for (const machine of result) {
        incoming.set(machine.id, []);
        outgoing.set(machine.id, []);
    }
    for (const conn of connections) {
        if (!incoming.has(conn.targetMachineId) || !outgoing.has(conn.sourceMachineId)) continue;
        incoming.get(conn.targetMachineId)!.push(conn.sourceMachineId);
        outgoing.get(conn.sourceMachineId)!.push(conn.targetMachineId);
    }

    const sources = result.filter((m) => (incoming.get(m.id)?.length || 0) === 0 && (outgoing.get(m.id)?.length || 0) > 0);
    const mids = result.filter((m) => (incoming.get(m.id)?.length || 0) > 0 && (outgoing.get(m.id)?.length || 0) > 0);
    const sinks = result.filter((m) => (incoming.get(m.id)?.length || 0) > 0 && (outgoing.get(m.id)?.length || 0) === 0);

    if (sources.length === 0 || mids.length === 0 || sinks.length === 0) return null;
    if (sources.length + mids.length + sinks.length !== result.length) return null;

    const sourceSet = new Set(sources.map((m) => m.id));
    const midSet = new Set(mids.map((m) => m.id));
    const sinkSet = new Set(sinks.map((m) => m.id));
    for (const conn of connections) {
        if (sourceSet.has(conn.sourceMachineId) && midSet.has(conn.targetMachineId)) continue;
        if (midSet.has(conn.sourceMachineId) && sinkSet.has(conn.targetMachineId)) continue;
        return null;
    }

    sources.sort((a, b) => (initialOrder.get(a.id) || 0) - (initialOrder.get(b.id) || 0));
    const sourceOrder = new Map(sources.map((m, idx) => [m.id, idx]));
    const isBalancedThreeLayer = sources.length === mids.length && mids.length === sinks.length;
    if (isBalancedThreeLayer) {
        mids.sort((a, b) => (initialOrder.get(a.id) || 0) - (initialOrder.get(b.id) || 0));
        sinks.sort((a, b) => (initialOrder.get(a.id) || 0) - (initialOrder.get(b.id) || 0));
    } else {
        mids.sort((a, b) => {
            const aOrder = averageNeighborOrder(incoming.get(a.id) || [], sourceOrder, sourceOrder.size);
            const bOrder = averageNeighborOrder(incoming.get(b.id) || [], sourceOrder, sourceOrder.size);
            if (aOrder !== bOrder) return aOrder - bOrder;
            return (initialOrder.get(a.id) || 0) - (initialOrder.get(b.id) || 0);
        });
        const midOrder = new Map(mids.map((m, idx) => [m.id, idx]));
        sinks.sort((a, b) => {
            const aOrder = averageNeighborOrder(incoming.get(a.id) || [], midOrder, midOrder.size);
            const bOrder = averageNeighborOrder(incoming.get(b.id) || [], midOrder, midOrder.size);
            if (aOrder !== bOrder) return aOrder - bOrder;
            return (initialOrder.get(a.id) || 0) - (initialOrder.get(b.id) || 0);
        });
    }

    const topH = rowHeight(sources);
    const midH = rowHeight(mids);
    const bottomH = rowHeight(sinks);
    const yTop = 1;
    const yMid = yTop + topH + 2;
    const yBottom = yMid + midH + 2;
    if (yBottom + bottomH > gridH) return null;

    if (!placeRowCenteredWithGap(sources, yTop, gridW, 1)) return null;
    if (!placeRowCenteredWithGap(mids, yMid, gridW, 1)) return null;
    if (!placeRowCenteredWithGap(sinks, yBottom, gridW, 1)) return null;

    return isCompletePlacementValid(result, gridW, gridH) ? result : null;
}

function ringChordSeed(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
): Machine[] | null {
    if (machines.length < 8) return null;
    if (connections.length < machines.length + Math.floor(machines.length / 3)) return null;

    const firstType = machines[0].type;
    if (machines.some((m) => m.type !== firstType)) return null;

    const result = machines.map((m) => ({ ...m, orientation: Orientation.NORTH }));
    const byId = new Map(result.map((m) => [m.id, m]));
    const indegree = new Map<string, number>();
    const outdegree = new Map<string, number>();
    const outgoing = new Map<string, Map<string, number>>();

    for (const machine of result) {
        indegree.set(machine.id, 0);
        outdegree.set(machine.id, 0);
        outgoing.set(machine.id, new Map());
    }
    for (const conn of connections) {
        if (!byId.has(conn.sourceMachineId) || !byId.has(conn.targetMachineId)) continue;
        outdegree.set(conn.sourceMachineId, (outdegree.get(conn.sourceMachineId) || 0) + 1);
        indegree.set(conn.targetMachineId, (indegree.get(conn.targetMachineId) || 0) + 1);
        const row = outgoing.get(conn.sourceMachineId)!;
        row.set(conn.targetMachineId, (row.get(conn.targetMachineId) || 0) + 1);
    }
    for (const machine of result) {
        if ((indegree.get(machine.id) || 0) === 0 || (outdegree.get(machine.id) || 0) === 0) {
            return null;
        }
    }

    const degreeScore = (id: string): number => (indegree.get(id) || 0) + (outdegree.get(id) || 0);
    const ids = result.map((m) => m.id);
    const unvisited = new Set(ids);
    const order: string[] = [];

    let current = ids[0];
    for (const id of ids) {
        if (degreeScore(id) > degreeScore(current)) current = id;
    }

    while (unvisited.size > 0) {
        if (!unvisited.has(current)) {
            current = Array.from(unvisited).sort((a, b) => degreeScore(b) - degreeScore(a) || a.localeCompare(b))[0];
        }
        order.push(current);
        unvisited.delete(current);
        if (unvisited.size === 0) break;

        const neighbors = outgoing.get(current);
        let next: string | null = null;
        let bestWeight = -1;
        if (neighbors) {
            for (const [nid, weight] of neighbors) {
                if (!unvisited.has(nid)) continue;
                const weighted = weight * 10 + degreeScore(nid);
                if (weighted > bestWeight) {
                    bestWeight = weighted;
                    next = nid;
                }
            }
        }

        if (!next) {
            next = Array.from(unvisited).sort((a, b) => degreeScore(b) - degreeScore(a) || a.localeCompare(b))[0];
        }
        current = next;
    }

    if (order.length !== result.length) return null;

    const topCount = Math.ceil(order.length / 2);
    const topRow = order.slice(0, topCount).map((id) => byId.get(id)!);
    const bottomRow = order.slice(topCount).reverse().map((id) => byId.get(id)!);

    const topH = rowHeight(topRow);
    const bottomH = rowHeight(bottomRow);
    const yTop = 1;
    let yBottom = yTop + topH + Math.max(4, Math.floor(Math.max(topH, bottomH) * 1.4));
    if (yBottom + bottomH > gridH) yBottom = gridH - bottomH - 1;
    if (yBottom <= yTop + topH) return null;

    if (!placeRowCentered(topRow, yTop, gridW)) return null;
    if (!placeRowCentered(bottomRow, yBottom, gridW)) return null;

    return isCompletePlacementValid(result, gridW, gridH) ? result : null;
}

function rowHeight(row: Machine[]): number {
    let maxHeight = 0;
    for (const machine of row) {
        maxHeight = Math.max(maxHeight, getOrientedDimensions(machine).height);
    }
    return maxHeight;
}

function averageNeighborOrder(
    neighbors: string[],
    order: Map<string, number>,
    fallback: number,
): number {
    let total = 0;
    let count = 0;
    for (const id of neighbors) {
        const idx = order.get(id);
        if (idx === undefined) continue;
        total += idx;
        count++;
    }
    return count > 0 ? total / count : fallback;
}

function isCompletePlacementValid(
    machines: Machine[],
    gridW: number,
    gridH: number,
): boolean {
    const allIds = new Set(machines.map((m) => m.id));
    for (const machine of machines) {
        if (!isValidPlacement(machine, machines, allIds, gridW, gridH)) return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────
// GEOMETRY UTILS
// ─────────────────────────────────────────────────────────

function isValidPlacement(
    machine: Machine,
    allMachines: Machine[],
    placed: Set<string>,
    gridW: number,
    gridH: number,
): boolean {
    const dims = getOrientedDimensions(machine);
    if (machine.x < 0 || machine.y < 0) return false;
    if (machine.x + dims.width > gridW || machine.y + dims.height > gridH) return false;

    for (const other of allMachines) {
        if (other.id === machine.id || !placed.has(other.id)) continue;
        if (machinesOverlap(machine, other)) return false;
    }
    return true;
}

function machinesOverlap(a: Machine, b: Machine): boolean {
    const aD = getOrientedDimensions(a);
    const bD = getOrientedDimensions(b);
    return !(
        a.x + aD.width <= b.x ||
        b.x + bD.width <= a.x ||
        a.y + aD.height <= b.y ||
        b.y + bD.height <= a.y
    );
}

function spiralSearch(
    cx: number,
    cy: number,
    machine: Machine,
    allMachines: Machine[],
    placed: Set<string>,
    gridW: number,
    gridH: number,
): { x: number; y: number } | null {
    for (let radius = 1; radius < Math.max(gridW, gridH); radius++) {
        for (let dx = -radius; dx <= radius; dx++) {
            for (let dy = -radius; dy <= radius; dy++) {
                if (Math.abs(dx) !== radius && Math.abs(dy) !== radius) continue;
                machine.x = cx + dx;
                machine.y = cy + dy;
                if (isValidPlacement(machine, allMachines, placed, gridW, gridH)) {
                    return { x: machine.x, y: machine.y };
                }
            }
        }
    }
    return null;
}

function buildLayoutFingerprint(machines: Machine[]): string {
    return machines
        .slice()
        .sort((a, b) => a.id.localeCompare(b.id))
        .map((m) => `${m.id}:${m.x},${m.y},${m.orientation}`)
        .join('|');
}

function layoutDistance(a: Machine[], b: Machine[]): number {
    if (a.length === 0 || b.length === 0) return Infinity;
    const otherById = new Map(b.map((machine) => [machine.id, machine]));
    let total = 0;
    let compared = 0;
    for (const machine of a) {
        const other = otherById.get(machine.id);
        if (!other) continue;
        total += Math.abs(machine.x - other.x);
        total += Math.abs(machine.y - other.y);
        total += machine.orientation === other.orientation ? 0 : 1;
        compared++;
    }
    if (compared === 0) return Infinity;
    return total / compared;
}

// ─────────────────────────────────────────────────────────
// ORIENTATION OPTIMIZATION (A*-verified)
// ─────────────────────────────────────────────────────────

/**
 * For each machine, try all 4 orientations and pick the one that
 * gives the best ACTUAL layout score (using full A* pathfinding).
 * Falls back to Manhattan if A* fails.
 */
function optimizeOrientationsUsingAStar(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
    fixedMachines: Map<string, Machine> = new Map(),
): Machine[] {
    const result = machines.map((m) => ({ ...m }));
    enforceFixedMachines(result, fixedMachines);

    // Score the current layout
    let currentBest = scorePlacement(result, connections, gridW, gridH, false);
    if (currentBest === Infinity) {
        currentBest = scorePlacement(result, connections, gridW, gridH, true);
    }

    // Try rotating each machine and keep improvement
    for (const machine of result) {
        if (fixedMachines.has(machine.id)) continue;
        const origOrient = machine.orientation;
        let bestOrient = origOrient;
        let bestScore = currentBest;

        for (const orient of ORIENTATIONS) {
            if (orient === origOrient) continue;
            machine.orientation = orient;
            const dims = getOrientedDimensions(machine);
            if (machine.x + dims.width > gridW || machine.y + dims.height > gridH) continue;

            // Check overlaps
            let valid = true;
            for (const other of result) {
                if (other.id === machine.id) continue;
                if (machinesOverlap(machine, other)) { valid = false; break; }
            }
            if (!valid) continue;

            const score = scorePlacement(result, connections, gridW, gridH, false);
            if (score < bestScore) {
                bestScore = score;
                bestOrient = orient;
            }
        }

        machine.orientation = bestOrient;
        currentBest = bestScore;
    }

    return result;
}

// ─────────────────────────────────────────────────────────
// SA ENGINE
// ─────────────────────────────────────────────────────────

interface SAConfig {
    initialTemp: number;
    coolingRate: number;
    minTemp: number;
    iterPerTemp: number;
    batchSize: number;
    useFastScore: boolean;
    operators?: SAOperatorConfig;
    random?: RandomFn;
    shouldStop?: () => boolean;
    maxIterations?: number;
    fixedMachineIds?: Set<string>;
}

function runSAAsync(
    startMachines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
    config: SAConfig,
    onProgress?: (best: ScoreBreakdown, iteration: number) => void,
): Promise<Machine[]> {
    return new Promise((resolve) => {
        let current = startMachines.map((m) => ({ ...m }));
        const fixedMachineIds = config.fixedMachineIds ?? new Set<string>();
        if (fixedMachineIds.size > 0) {
            const fixedSnapshots = buildFixedMachineMap(current);
            enforceFixedMachines(current, fixedSnapshots);
        }
        let currentScore = scorePlacement(current, connections, gridW, gridH, config.useFastScore);
        let best = current.map((m) => ({ ...m }));
        let bestScore = currentScore;
        let temp = config.initialTemp;
        let iter = 0;
        let stagnation = 0;
        let iterationsSinceBest = 0;
        let largeMoveCooldownRemaining = 0;
        const random = config.random ?? Math.random;
        const operatorConfig = config.operators ?? {
            largeMoveRate: 0.15,
            clusterMoveMinSize: 2,
            clusterMoveMaxSize: 4,
            adaptiveOps: false,
            adaptiveWindow: 80,
            adaptiveWarmupIterations: 0,
            adaptiveMaxOperatorProb: 1,
            adaptiveStagnationResetWindow: Number.MAX_SAFE_INTEGER,
            adaptiveFlattenFactor: 0,
            largeMoveRateEarly: 0.15,
            largeMoveRateLate: 0.15,
            largeMoveCooldownAfterImprove: 0,
            criticalNetRate: 0,
            repairBeamWidth: 1,
        };
        const operators = buildMoveOperators(operatorConfig);
        const operatorStats = new Map<MoveOperatorId, {
            attempts: number;
            accepted: number;
            improved: number;
            recentGain: number[];
        }>();
        for (const operator of operators) {
            operatorStats.set(operator.id, {
                attempts: 0,
                accepted: 0,
                improved: 0,
                recentGain: [],
            });
        }
        const adaptiveWindow = Math.max(20, operatorConfig.adaptiveWindow);
        const rewardDecay = 0.9;
        const majorImproveThreshold = (score: number): number => Math.max(6, score * 0.0125);
        let activeOperatorConfig = operatorConfig;
        const pickOperator = (): MoveOperator => {
            activeOperatorConfig = buildAdaptiveOperatorConfig(
                operatorConfig,
                temp,
                config.initialTemp,
                config.minTemp,
                iterationsSinceBest,
                largeMoveCooldownRemaining,
            );
            const activeOperators = buildMoveOperators(activeOperatorConfig);
            const isAdaptationWarm = operatorConfig.adaptiveOps && iter >= operatorConfig.adaptiveWarmupIterations;
            const baselineWeights = activeOperators.map((operator) => operator.baseWeight);
            const floors = activeOperators.map((operator) => operator.minProbability);

            let weighted = baselineWeights.slice();
            if (isAdaptationWarm) {
                weighted = activeOperators.map((operator, idx) => {
                    const stats = operatorStats.get(operator.id);
                    const reward = stats ? computeDecayedPositiveGain(stats.recentGain, rewardDecay) : 0;
                    const rewardBoost = 1 + Math.log1p(Math.max(0, reward));
                    return baselineWeights[idx] * rewardBoost;
                });
            }

            let probabilities = normalizeProbabilitiesWithFloorAndCap(
                weighted,
                floors,
                isAdaptationWarm ? operatorConfig.adaptiveMaxOperatorProb : 1,
            );

            if (
                isAdaptationWarm
                && operatorConfig.adaptiveStagnationResetWindow > 0
                && iterationsSinceBest >= operatorConfig.adaptiveStagnationResetWindow
                && operatorConfig.adaptiveFlattenFactor > 0
            ) {
                const baselineProbabilities = normalizeProbabilitiesWithFloorAndCap(baselineWeights, floors, 1);
                probabilities = probabilities.map((value, idx) => (
                    value * (1 - operatorConfig.adaptiveFlattenFactor)
                    + baselineProbabilities[idx] * operatorConfig.adaptiveFlattenFactor
                ));
                probabilities = normalizePositiveWeights(probabilities);
            }

            const selectedIndex = pickProbabilityIndex(probabilities, random);
            return activeOperators[selectedIndex] ?? activeOperators[activeOperators.length - 1];
        };
        const shouldStop = (): boolean => {
            if (config.shouldStop && config.shouldStop()) return true;
            if (config.maxIterations !== undefined && iter >= config.maxIterations) return true;
            return false;
        };

        function step() {
            if (shouldStop()) {
                resolve(best);
                return;
            }

            const prevBest = bestScore;

            saLoop:
            for (let b = 0; b < config.batchSize && temp > config.minTemp; b++) {
                for (let i = 0; i < config.iterPerTemp; i++) {
                    if (shouldStop()) break saLoop;
                    iter++;
                    const chosenOperator = pickOperator();
                    const preMoveScore = currentScore;
                    const candidate = perturbSmart(
                        current,
                        connections,
                        gridW,
                        gridH,
                        random,
                        activeOperatorConfig,
                        chosenOperator.id,
                        fixedMachineIds,
                    );
                    const candidateScore = scorePlacement(candidate, connections, gridW, gridH, config.useFastScore);
                    const stats = operatorStats.get(chosenOperator.id)!;
                    stats.attempts++;
                    let gain = 0;
                    let improvedBest = false;

                    if (candidateScore < Infinity) {
                        const delta = candidateScore - currentScore;
                        if (delta < 0 || random() < Math.exp(-delta / temp)) {
                            current = candidate;
                            currentScore = candidateScore;
                            gain = Math.max(0, preMoveScore - candidateScore);
                            stats.accepted++;
                            if (candidateScore < bestScore) {
                                const bestImprovement = bestScore - candidateScore;
                                bestScore = candidateScore;
                                best = candidate.map((m) => ({ ...m }));
                                stats.improved++;
                                improvedBest = true;
                                if (
                                    operatorConfig.largeMoveCooldownAfterImprove > 0
                                    && bestImprovement >= majorImproveThreshold(bestScore)
                                ) {
                                    largeMoveCooldownRemaining = operatorConfig.largeMoveCooldownAfterImprove;
                                }
                            }
                        }
                    }

                    stats.recentGain.push(gain);
                    if (stats.recentGain.length > adaptiveWindow) stats.recentGain.shift();
                    iterationsSinceBest = improvedBest ? 0 : iterationsSinceBest + 1;
                    if (largeMoveCooldownRemaining > 0) {
                        largeMoveCooldownRemaining--;
                    }
                }
                temp *= config.coolingRate;
            }

            // Track stagnation
            if (bestScore >= prevBest) {
                stagnation++;
            } else {
                stagnation = 0;
            }

            if (onProgress) {
                const breakdown = buildAndScore(best, connections, gridW, gridH);
                if (breakdown) {
                    onProgress(breakdown.score, iter);
                } else {
                    // A* routing failed — report proxy score components.
                    const fastBreakdown = computeFastScoreBreakdown(
                        best,
                        connections,
                        new Map(best.map((m) => [m.id, m])),
                    );
                    if (fastBreakdown.totalScore < Infinity) {
                        onProgress(fastBreakdown, iter);
                    }
                }
            }

            // Reheat if stagnating
            if (stagnation > 5 && temp > config.minTemp) {
                temp = Math.min(config.initialTemp * 0.5, temp * 3);
                current = best.map((m) => ({ ...m }));
                currentScore = bestScore;
                stagnation = 0;
            }

            if (temp > config.minTemp && !shouldStop()) {
                setTimeout(step, 0);
            } else {
                resolve(best);
            }
        }

        setTimeout(step, 0);
    });
}

function buildAdaptiveOperatorConfig(
    base: SAOperatorConfig,
    temp: number,
    initialTemp: number,
    minTemp: number,
    iterationsSinceBest: number,
    largeMoveCooldownRemaining: number,
): SAOperatorConfig {
    const span = Math.max(1e-6, initialTemp - minTemp);
    const normalizedTemp = Math.max(0, Math.min(1, (temp - minTemp) / span));
    const inEarlyPhase = normalizedTemp >= 0.45;
    let scheduledLargeRate = inEarlyPhase ? base.largeMoveRateEarly : base.largeMoveRateLate;
    if (iterationsSinceBest > Math.max(30, Math.floor(base.adaptiveStagnationResetWindow * 0.6))) {
        scheduledLargeRate = Math.max(scheduledLargeRate, base.largeMoveRateEarly);
    }
    if (largeMoveCooldownRemaining > 0) {
        scheduledLargeRate = 0;
    }
    const divisor = Math.max(1e-6, Math.max(base.largeMoveRate, base.criticalNetRate));
    const criticalShare = Math.max(0, Math.min(1, base.criticalNetRate / divisor));
    const criticalNetRate = scheduledLargeRate * criticalShare;

    return {
        ...base,
        largeMoveRate: scheduledLargeRate,
        criticalNetRate,
    };
}

function computeDecayedPositiveGain(gains: number[], decay: number): number {
    if (gains.length === 0) return 0;
    let weightedSum = 0;
    let weightTotal = 0;
    let weight = 1;
    for (let i = gains.length - 1; i >= 0; i--) {
        const value = Math.max(0, gains[i]);
        weightedSum += value * weight;
        weightTotal += weight;
        weight *= decay;
    }
    return weightTotal > 0 ? weightedSum / weightTotal : 0;
}

function normalizePositiveWeights(values: number[]): number[] {
    if (values.length === 0) return [];
    const positives = values.map((value) => (Number.isFinite(value) && value > 0 ? value : 0));
    const total = positives.reduce((sum, value) => sum + value, 0);
    if (total <= 1e-12) {
        return positives.map(() => 1 / positives.length);
    }
    return positives.map((value) => value / total);
}

function normalizeProbabilitiesWithFloorAndCap(
    rawWeights: number[],
    minProbabilities: number[],
    maxProbability: number,
): number[] {
    const count = rawWeights.length;
    if (count === 0) return [];
    const cap = Math.max(1 / count, Math.min(1, maxProbability));
    const floors = minProbabilities.map((value) => Math.max(0, Math.min(cap, value)));
    const floorTotal = floors.reduce((sum, value) => sum + value, 0);
    if (floorTotal >= 1 - 1e-9) {
        return normalizePositiveWeights(floors);
    }

    const normalizedRaw = normalizePositiveWeights(rawWeights);
    const probabilities = floors.slice();
    let remaining = Math.max(0, 1 - floorTotal);
    let free = Array.from({ length: count }, (_, idx) => idx).filter((idx) => probabilities[idx] < cap - 1e-9);
    let guard = 0;

    while (remaining > 1e-9 && free.length > 0 && guard < count * 6) {
        guard++;
        const freeWeightSum = free.reduce((sum, idx) => sum + normalizedRaw[idx], 0);
        const usingUniform = freeWeightSum <= 1e-12;
        const denominator = usingUniform ? free.length : freeWeightSum;

        let consumed = 0;
        let capped = false;
        for (const idx of free) {
            const weight = usingUniform ? 1 : normalizedRaw[idx];
            const share = remaining * (weight / denominator);
            const room = cap - probabilities[idx];
            const add = Math.min(room, share);
            probabilities[idx] += add;
            consumed += add;
            if (room - add <= 1e-9) capped = true;
        }

        if (consumed <= 1e-12) break;
        remaining = Math.max(0, 1 - probabilities.reduce((sum, value) => sum + value, 0));
        if (!capped) break;
        free = free.filter((idx) => probabilities[idx] < cap - 1e-9);
    }

    if (remaining > 1e-9) {
        const candidates = Array.from({ length: count }, (_, idx) => idx)
            .filter((idx) => probabilities[idx] < cap - 1e-9);
        if (candidates.length > 0) {
            const even = remaining / candidates.length;
            for (const idx of candidates) {
                probabilities[idx] += Math.min(cap - probabilities[idx], even);
            }
        }
    }

    return normalizePositiveWeights(probabilities);
}

function pickProbabilityIndex(probabilities: number[], random: RandomFn): number {
    const normalized = normalizePositiveWeights(probabilities);
    const target = random();
    let running = 0;
    for (let i = 0; i < normalized.length; i++) {
        running += normalized[i];
        if (target <= running) return i;
    }
    return Math.max(0, normalized.length - 1);
}

/**
 * Score a machine placement.
 * When useFast=false and A* routing fails, falls back to fast scoring
 * with a penalty instead of returning Infinity. This prevents the SA
 * from getting permanently stuck in unroutable states.
 */
function scorePlacement(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
    useFast: boolean,
): number {
    // Validity check — overlaps and bounds are always Infinity
    for (let i = 0; i < machines.length; i++) {
        const dims = getOrientedDimensions(machines[i]);
        if (machines[i].x < 0 || machines[i].y < 0) return Infinity;
        if (machines[i].x + dims.width > gridW || machines[i].y + dims.height > gridH) return Infinity;
        for (let j = i + 1; j < machines.length; j++) {
            if (machinesOverlap(machines[i], machines[j])) return Infinity;
        }
    }

    const machineMap = new Map(machines.map((m) => [m.id, m]));

    if (useFast) {
        return computeFastScore(machines, connections, machineMap);
    } else {
        const result = buildGrid(machines, connections, gridW, gridH);
        if (result) {
            return evaluateGrid(result).totalScore;
        }
        // A* routing failed — fall back to fast scoring with a substantial
        // penalty so SA does not camp in unroutable regions.
        const fastScore = computeFastScore(machines, connections, machineMap);
        if (fastScore === Infinity) return Infinity;
        const unroutablePenalty =
            UNROUTABLE_BASE_PENALTY
            + connections.length * UNROUTABLE_PER_CONNECTION_PENALTY
            + machines.length * UNROUTABLE_PER_MACHINE_PENALTY;
        return fastScore + unroutablePenalty;
    }
}

/** Compute fast Manhattan-based score without A* routing */
function computeFastScore(
    machines: Machine[],
    connections: Connection[],
    machineMap: Map<string, Machine>,
): number {
    return computeFastScoreBreakdown(machines, connections, machineMap).totalScore;
}

function computeFastScoreBreakdown(
    machines: Machine[],
    connections: Connection[],
    machineMap: Map<string, Machine>,
): ScoreBreakdown {
    let totalDist = 0;
    let cornerCount = 0;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    const includeCell = (x: number, y: number): void => {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
    };
    const includeMachine = (machine: Machine): void => {
        const dims = getOrientedDimensions(machine);
        includeCell(machine.x, machine.y);
        includeCell(machine.x + dims.width - 1, machine.y + dims.height - 1);
    };

    for (const machine of machines) {
        includeMachine(machine);
    }

    for (const conn of connections) {
        const src = machineMap.get(conn.sourceMachineId);
        const tgt = machineMap.get(conn.targetMachineId);
        if (!src || !tgt) {
            return { totalBelts: Infinity, boundingBoxArea: Infinity, cornerCount: Infinity, totalScore: Infinity };
        }
        const srcPorts = getMachinePorts(src);
        const tgtPorts = getMachinePorts(tgt);
        const srcPort = srcPorts.outputs[conn.sourcePortIndex];
        const tgtPort = tgtPorts.inputs[conn.targetPortIndex];
        if (!srcPort || !tgtPort) {
            return { totalBelts: Infinity, boundingBoxArea: Infinity, cornerCount: Infinity, totalScore: Infinity };
        }
        const srcTile = getPortExternalTile(srcPort);
        const tgtTile = getPortExternalTile(tgtPort);
        includeCell(srcTile.x, srcTile.y);
        includeCell(tgtTile.x, tgtTile.y);
        const dx = Math.abs(srcTile.x - tgtTile.x);
        const dy = Math.abs(srcTile.y - tgtTile.y);
        totalDist += dx + dy;
        if (dx > 0 && dy > 0) {
            cornerCount++;
        }
    }
    const boundingBoxArea =
        minX <= maxX && minY <= maxY
            ? (maxX - minX + 1) * (maxY - minY + 1)
            : 0;

    const totalScore =
        totalDist * BELT_WEIGHT
        + boundingBoxArea * AREA_WEIGHT
        + cornerCount * CORNER_WEIGHT;

    return {
        totalBelts: totalDist,
        boundingBoxArea,
        cornerCount,
        totalScore,
    };
}

// ─────────────────────────────────────────────────────────
// SMART PERTURBATION
// ─────────────────────────────────────────────────────────

function buildMoveOperators(operatorConfig?: SAOperatorConfig): MoveOperator[] {
    const largeMoveRate = Math.max(0, Math.min(0.6, operatorConfig?.largeMoveRate ?? 0.15));
    const criticalNetRate = Math.max(0, Math.min(largeMoveRate, operatorConfig?.criticalNetRate ?? 0));
    const clusterRate = Math.max(0, largeMoveRate - criticalNetRate);
    const clusterFloor = clusterRate > 1e-9 ? Math.min(0.03, clusterRate * 0.35) : 0;
    const criticalFloor = criticalNetRate > 1e-9 ? Math.min(0.02, criticalNetRate * 0.35) : 0;
    const sharedScale = Math.max(0.05, 1 - largeMoveRate);
    return [
        { id: 'move_toward_neighbor', baseWeight: 0.2 * sharedScale, minProbability: 0.02 },
        { id: 'move_to_source', baseWeight: 0.12 * sharedScale, minProbability: 0.02 },
        { id: 'port_facing_jump', baseWeight: 0.14 * sharedScale, minProbability: 0.02 },
        { id: 'try_different_port', baseWeight: 0.12 * sharedScale, minProbability: 0.02 },
        { id: 'random_shift', baseWeight: 0.13 * sharedScale, minProbability: 0.02 },
        { id: 'swap_positions', baseWeight: 0.11 * sharedScale, minProbability: 0.02 },
        { id: 'rotate_best', baseWeight: 0.09 * sharedScale, minProbability: 0.02 },
        { id: 'joint_move_rotate', baseWeight: 0.09 * sharedScale, minProbability: 0.02 },
        {
            id: 'cluster_destroy_repair',
            baseWeight: clusterRate,
            minProbability: clusterFloor,
        },
        {
            id: 'critical_net_focus',
            baseWeight: criticalNetRate,
            minProbability: criticalFloor,
        },
    ];
}

function pickMoveOperator(operators: MoveOperator[], random: RandomFn): MoveOperator {
    const totalWeight = operators.reduce((sum, op) => sum + op.baseWeight, 0);
    const roll = random() * (totalWeight || 1);
    let cumulative = 0;
    for (const operator of operators) {
        cumulative += operator.baseWeight;
        if (roll <= cumulative) return operator;
    }
    return operators[operators.length - 1];
}

function perturbSmart(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
    random: RandomFn = Math.random,
    operatorConfig?: SAOperatorConfig,
    forcedOperator?: MoveOperatorId,
    fixedMachineIds: Set<string> = new Set(),
): Machine[] {
    const result = machines.map((m) => ({ ...m }));
    const movableIndexes = getMovableMachineIndexes(result, fixedMachineIds);
    if (movableIndexes.length === 0) return result;
    const operators = buildMoveOperators(operatorConfig);
    const operator = forcedOperator
        ? operators.find((item) => item.id === forcedOperator) ?? operators[0]
        : pickMoveOperator(operators, random);
    const clusterMinSize = Math.min(operatorConfig?.clusterMoveMinSize ?? 2, operatorConfig?.clusterMoveMaxSize ?? 5);
    const clusterMaxSize = Math.max(operatorConfig?.clusterMoveMinSize ?? 2, operatorConfig?.clusterMoveMaxSize ?? 5);
    const repairBeamWidth = Math.max(1, Math.floor(operatorConfig?.repairBeamWidth ?? 1));

    switch (operator.id) {
        case 'move_toward_neighbor':
            moveTowardNeighbor(result, connections, gridW, gridH, random, fixedMachineIds);
            break;
        case 'move_to_source':
            moveToSource(result, connections, gridW, gridH, random, fixedMachineIds);
            break;
        case 'port_facing_jump':
            portFacingJump(result, connections, gridW, gridH, random, fixedMachineIds);
            break;
        case 'try_different_port':
            tryDifferentPortAssignment(result, connections, random, fixedMachineIds);
            break;
        case 'random_shift': {
            const idx = movableIndexes[Math.floor(random() * movableIndexes.length)];
            const shift = Math.floor(random() * 3) + 1;
            const dir = Math.floor(random() * 4);
            switch (dir) {
                case 0: result[idx].x += shift; break;
                case 1: result[idx].x -= shift; break;
                case 2: result[idx].y += shift; break;
                case 3: result[idx].y -= shift; break;
            }
            const dims = getOrientedDimensions(result[idx]);
            result[idx].x = Math.max(0, Math.min(gridW - dims.width, result[idx].x));
            result[idx].y = Math.max(0, Math.min(gridH - dims.height, result[idx].y));
            break;
        }
        case 'swap_positions':
            if (movableIndexes.length > 1) {
                const i = movableIndexes[Math.floor(random() * movableIndexes.length)];
                const otherIndexes = movableIndexes.filter((idx) => idx !== i);
                const j = otherIndexes[Math.floor(random() * otherIndexes.length)];
                const tmpX = result[i].x;
                const tmpY = result[i].y;
                result[i].x = result[j].x;
                result[i].y = result[j].y;
                result[j].x = tmpX;
                result[j].y = tmpY;
            }
            break;
        case 'rotate_best': {
            const idx = movableIndexes[Math.floor(random() * movableIndexes.length)];
            const machineMap = new Map(result.map((m) => [m.id, m]));
            let bestOrient = result[idx].orientation;
            let bestDist = Infinity;
            for (const orient of ORIENTATIONS) {
                result[idx].orientation = orient;
                const dims = getOrientedDimensions(result[idx]);
                if (result[idx].x + dims.width > gridW || result[idx].y + dims.height > gridH) continue;
                let totalDist = 0;
                for (const conn of connections) {
                    if (conn.sourceMachineId !== result[idx].id && conn.targetMachineId !== result[idx].id) continue;
                    const src = machineMap.get(conn.sourceMachineId);
                    const tgt = machineMap.get(conn.targetMachineId);
                    if (!src || !tgt) continue;
                    const sp = getMachinePorts(src).outputs[conn.sourcePortIndex];
                    const tp = getMachinePorts(tgt).inputs[conn.targetPortIndex];
                    if (sp && tp) totalDist += estimateBeltLength(sp, tp);
                }
                if (totalDist < bestDist) {
                    bestDist = totalDist;
                    bestOrient = orient;
                }
            }
            result[idx].orientation = bestOrient;
            break;
        }
        case 'joint_move_rotate': {
            const idx = movableIndexes[Math.floor(random() * movableIndexes.length)];
            const shift = Math.floor(random() * 2) + 1;
            const dir = Math.floor(random() * 4);
            switch (dir) {
                case 0: result[idx].x += shift; break;
                case 1: result[idx].x -= shift; break;
                case 2: result[idx].y += shift; break;
                case 3: result[idx].y -= shift; break;
            }
            const dims = getOrientedDimensions(result[idx]);
            result[idx].x = Math.max(0, Math.min(gridW - dims.width, result[idx].x));
            result[idx].y = Math.max(0, Math.min(gridH - dims.height, result[idx].y));
            result[idx].orientation = ORIENTATIONS[Math.floor(random() * 4)];
            break;
        }
        case 'cluster_destroy_repair':
            applyRepairBeamCandidates(
                result,
                connections,
                gridW,
                gridH,
                repairBeamWidth,
                random,
                (candidate, candidateRandom) => applyDestroyRepairCluster(
                    candidate,
                    connections,
                    gridW,
                    gridH,
                    clusterMinSize,
                    clusterMaxSize,
                    candidateRandom,
                    fixedMachineIds,
                ),
            );
            break;
        case 'critical_net_focus':
            applyRepairBeamCandidates(
                result,
                connections,
                gridW,
                gridH,
                repairBeamWidth,
                random,
                (candidate, candidateRandom) => applyCriticalNetFocusedMove(
                    candidate,
                    connections,
                    gridW,
                    gridH,
                    clusterMinSize,
                    clusterMaxSize,
                    candidateRandom,
                    fixedMachineIds,
                ),
            );
            break;
    }

    return result;
}

function getMovableMachineIndexes(machines: Machine[], fixedMachineIds: Set<string>): number[] {
    const movable: number[] = [];
    for (let i = 0; i < machines.length; i++) {
        if (fixedMachineIds.has(machines[i].id)) continue;
        movable.push(i);
    }
    return movable;
}

function applyDestroyRepairCluster(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
    minSize: number,
    maxSize: number,
    random: RandomFn,
    fixedMachineIds: Set<string>,
    forcedClusterIds?: string[],
): boolean {
    const movableIds = machines
        .map((machine) => machine.id)
        .filter((id) => !fixedMachineIds.has(id));
    if (movableIds.length < 2 || connections.length === 0) return false;

    const machineMap = new Map(machines.map((m) => [m.id, m]));
    const adjacency = buildAdjacencyWeights(connections);
    const clusterIds = forcedClusterIds && forcedClusterIds.length > 0
        ? Array.from(new Set(forcedClusterIds.filter((id) => machineMap.has(id) && !fixedMachineIds.has(id))))
        : pickConnectedCluster(machines, adjacency, minSize, maxSize, random, fixedMachineIds);
    if (clusterIds.length < 2) return false;

    const clusterSet = new Set(clusterIds);
    const original = new Map<string, Machine>();
    for (const id of clusterIds) {
        const machine = machineMap.get(id);
        if (machine) original.set(id, { ...machine });
    }

    const placed = new Set(machines.filter((m) => !clusterSet.has(m.id)).map((m) => m.id));
    const ordered = clusterIds.slice().sort((a, b) => {
        const aExternal = countExternalConnections(a, connections, clusterSet);
        const bExternal = countExternalConnections(b, connections, clusterSet);
        return bExternal - aExternal;
    });

    for (const id of ordered) {
        const machine = machineMap.get(id);
        if (!machine) continue;

        const candidate = findDestroyRepairPlacement(
            machine,
            machines,
            connections,
            placed,
            gridW,
            gridH,
            random,
        );
        if (!candidate) {
            for (const [restoreId, snapshot] of original) {
                const target = machineMap.get(restoreId);
                if (!target) continue;
                target.x = snapshot.x;
                target.y = snapshot.y;
                target.orientation = snapshot.orientation;
            }
            return false;
        }

        machine.x = candidate.x;
        machine.y = candidate.y;
        machine.orientation = candidate.orientation;
        placed.add(id);
    }

    return true;
}

function applyRepairBeamCandidates(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
    beamWidth: number,
    random: RandomFn,
    applyCandidate: (candidateMachines: Machine[], candidateRandom: RandomFn) => boolean,
): void {
    const original = machines.map((machine) => ({ ...machine }));
    const attempts = Math.max(1, Math.floor(beamWidth));
    let bestCandidate: Machine[] | null = null;
    let bestScore = Infinity;

    for (let attempt = 0; attempt < attempts; attempt++) {
        const candidate = original.map((machine) => ({ ...machine }));
        const candidateRandom = attempts === 1
            ? random
            : createSeededRandom(
                ((Math.floor(random() * 0x100000000) >>> 0) ^ ((attempt + 1) * 2654435761)) >>> 0,
            );
        const succeeded = applyCandidate(candidate, candidateRandom);
        if (!succeeded) continue;
        const candidateScore = scorePlacement(candidate, connections, gridW, gridH, false);
        if (!Number.isFinite(candidateScore)) continue;
        if (candidateScore < bestScore) {
            bestScore = candidateScore;
            bestCandidate = candidate;
        }
    }

    const selected = bestCandidate ?? original;
    for (let i = 0; i < machines.length && i < selected.length; i++) {
        machines[i].x = selected[i].x;
        machines[i].y = selected[i].y;
        machines[i].orientation = selected[i].orientation;
    }
}

function applyCriticalNetFocusedMove(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
    clusterMinSize: number,
    clusterMaxSize: number,
    random: RandomFn,
    fixedMachineIds: Set<string>,
): boolean {
    if (machines.length < 2 || connections.length === 0) return false;
    const machineMap = new Map(machines.map((machine) => [machine.id, machine]));
    const scoredConnections = connections
        .map((connection) => ({
            connection,
            pain: estimateConnectionPain(connection, machineMap),
        }))
        .filter((entry) => Number.isFinite(entry.pain) && entry.pain > 0)
        .sort((a, b) => b.pain - a.pain);
    if (scoredConnections.length === 0) return false;

    const topCount = Math.max(1, Math.floor(scoredConnections.length * 0.35));
    const targetConnection = scoredConnections[Math.floor(random() * topCount)].connection;
    const machinePain = new Map<string, number>();
    for (const entry of scoredConnections.slice(0, Math.max(topCount, 4))) {
        machinePain.set(
            entry.connection.sourceMachineId,
            (machinePain.get(entry.connection.sourceMachineId) ?? 0) + entry.pain,
        );
        machinePain.set(
            entry.connection.targetMachineId,
            (machinePain.get(entry.connection.targetMachineId) ?? 0) + entry.pain,
        );
    }

    const seedCluster = new Set<string>([
        targetConnection.sourceMachineId,
        targetConnection.targetMachineId,
    ]);
    for (const id of Array.from(seedCluster)) {
        if (fixedMachineIds.has(id)) seedCluster.delete(id);
    }
    const targetClusterSize = Math.max(
        2,
        Math.min(
            Math.max(2, clusterMinSize),
            Math.min(clusterMaxSize, 4),
        ),
    );
    const rankedMachineIds = Array.from(machinePain.entries())
        .sort((a, b) => b[1] - a[1])
        .map(([id]) => id);
    for (const id of rankedMachineIds) {
        if (seedCluster.size >= targetClusterSize) break;
        if (fixedMachineIds.has(id)) continue;
        seedCluster.add(id);
    }

    if (seedCluster.size >= 2) {
        const clustered = applyDestroyRepairCluster(
            machines,
            connections,
            gridW,
            gridH,
            seedCluster.size,
            seedCluster.size,
            random,
            fixedMachineIds,
            Array.from(seedCluster),
        );
        if (clustered) return true;
    }

    const fallbackOrder = [
        targetConnection.sourceMachineId,
        targetConnection.targetMachineId,
    ]
        .filter((id) => !fixedMachineIds.has(id))
        .sort((a, b) => (machinePain.get(b) ?? 0) - (machinePain.get(a) ?? 0));
    for (const machineId of fallbackOrder) {
        const machine = machineMap.get(machineId);
        if (!machine) continue;
        const before = { x: machine.x, y: machine.y, orientation: machine.orientation };
        const placed = new Set(machines.filter((item) => item.id !== machineId).map((item) => item.id));
        const candidate = findDestroyRepairPlacement(
            machine,
            machines,
            connections,
            placed,
            gridW,
            gridH,
            random,
        );
        if (!candidate) continue;
        machine.x = candidate.x;
        machine.y = candidate.y;
        machine.orientation = candidate.orientation;
        if (
            candidate.x !== before.x
            || candidate.y !== before.y
            || candidate.orientation !== before.orientation
        ) {
            return true;
        }
    }

    return false;
}

function estimateConnectionPain(
    connection: Connection,
    machineMap: Map<string, Machine>,
): number {
    const src = machineMap.get(connection.sourceMachineId);
    const tgt = machineMap.get(connection.targetMachineId);
    if (!src || !tgt) return Infinity;
    const srcPort = getMachinePorts(src).outputs[connection.sourcePortIndex];
    const tgtPort = getMachinePorts(tgt).inputs[connection.targetPortIndex];
    if (!srcPort || !tgtPort) return Infinity;
    const dx = Math.abs(srcPort.x - tgtPort.x);
    const dy = Math.abs(srcPort.y - tgtPort.y);
    const cornerProxy = dx > 0 && dy > 0 ? 1 : 0;
    return estimateBeltLength(srcPort, tgtPort) + cornerProxy * 2;
}

function countExternalConnections(id: string, connections: Connection[], cluster: Set<string>): number {
    let total = 0;
    for (const conn of connections) {
        if (conn.sourceMachineId === id && !cluster.has(conn.targetMachineId)) total++;
        if (conn.targetMachineId === id && !cluster.has(conn.sourceMachineId)) total++;
    }
    return total;
}

function buildAdjacencyWeights(connections: Connection[]): Map<string, Map<string, number>> {
    const adjacency = new Map<string, Map<string, number>>();
    const addEdge = (a: string, b: string): void => {
        if (!adjacency.has(a)) adjacency.set(a, new Map<string, number>());
        const neighbors = adjacency.get(a)!;
        neighbors.set(b, (neighbors.get(b) ?? 0) + 1);
    };
    for (const conn of connections) {
        addEdge(conn.sourceMachineId, conn.targetMachineId);
        addEdge(conn.targetMachineId, conn.sourceMachineId);
    }
    return adjacency;
}

function pickConnectedCluster(
    machines: Machine[],
    adjacency: Map<string, Map<string, number>>,
    minSize: number,
    maxSize: number,
    random: RandomFn,
    fixedMachineIds: Set<string>,
): string[] {
    const movableMachines = machines.filter((machine) => !fixedMachineIds.has(machine.id));
    if (movableMachines.length === 0) return [];

    const boundedMin = Math.max(2, Math.min(minSize, movableMachines.length));
    const boundedMax = Math.max(boundedMin, Math.min(maxSize, movableMachines.length));
    const targetSize = boundedMin + Math.floor(random() * (boundedMax - boundedMin + 1));

    const start = movableMachines[Math.floor(random() * movableMachines.length)].id;
    const selected = new Set<string>([start]);

    while (selected.size < targetSize) {
        const candidates = new Map<string, number>();
        for (const chosen of selected) {
            const neighbors = adjacency.get(chosen);
            if (!neighbors) continue;
            for (const [neighborId, weight] of neighbors) {
                if (selected.has(neighborId)) continue;
                if (fixedMachineIds.has(neighborId)) continue;
                candidates.set(neighborId, (candidates.get(neighborId) ?? 0) + weight);
            }
        }
        if (candidates.size === 0) break;

        const totalWeight = Array.from(candidates.values()).reduce((sum, value) => sum + value, 0);
        let roll = random() * (totalWeight || 1);
        let pickedId: string | null = null;
        for (const [id, weight] of candidates) {
            roll -= weight;
            if (roll <= 0) {
                pickedId = id;
                break;
            }
        }
        if (!pickedId) {
            pickedId = candidates.keys().next().value ?? null;
        }
        if (!pickedId) break;
        selected.add(pickedId);
    }

    return Array.from(selected);
}

function findDestroyRepairPlacement(
    machine: Machine,
    allMachines: Machine[],
    connections: Connection[],
    placed: Set<string>,
    gridW: number,
    gridH: number,
    random: RandomFn,
): { x: number; y: number; orientation: Orientation } | null {
    const machineMap = new Map(allMachines.map((m) => [m.id, m]));
    const connectedPlacedNeighbors = new Set<string>();
    for (const conn of connections) {
        if (conn.sourceMachineId === machine.id && placed.has(conn.targetMachineId)) {
            connectedPlacedNeighbors.add(conn.targetMachineId);
        }
        if (conn.targetMachineId === machine.id && placed.has(conn.sourceMachineId)) {
            connectedPlacedNeighbors.add(conn.sourceMachineId);
        }
    }

    const original = { x: machine.x, y: machine.y, orientation: machine.orientation };
    let bestX = original.x;
    let bestY = original.y;
    let bestOrientation = original.orientation;
    let bestCost = Infinity;
    let found = false;
    const tryPlacement = (x: number, y: number, orientation: Orientation): void => {
        machine.x = x;
        machine.y = y;
        machine.orientation = orientation;
        if (!isValidPlacement(machine, allMachines, placed, gridW, gridH)) return;
        const cost = estimateMachineConnectionCost(machine.id, allMachines, connections);
        if (!Number.isFinite(cost)) return;
        if (cost < bestCost) {
            bestCost = cost;
            bestX = x;
            bestY = y;
            bestOrientation = orientation;
            found = true;
        }
    };

    for (const neighborId of connectedPlacedNeighbors) {
        const neighbor = machineMap.get(neighborId);
        if (!neighbor) continue;
        const nDims = getOrientedDimensions(neighbor);
        for (const orient of ORIENTATIONS) {
            machine.orientation = orient;
            const dims = getOrientedDimensions(machine);
            const positions = [
                { x: neighbor.x, y: neighbor.y + nDims.height + 1 },
                { x: neighbor.x + Math.floor((nDims.width - dims.width) / 2), y: neighbor.y + nDims.height + 1 },
                { x: neighbor.x, y: neighbor.y - dims.height - 1 },
                { x: neighbor.x + Math.floor((nDims.width - dims.width) / 2), y: neighbor.y - dims.height - 1 },
                { x: neighbor.x + nDims.width + 1, y: neighbor.y },
                { x: neighbor.x + nDims.width + 1, y: neighbor.y + Math.floor((nDims.height - dims.height) / 2) },
                { x: neighbor.x - dims.width - 1, y: neighbor.y },
                { x: neighbor.x - dims.width - 1, y: neighbor.y + Math.floor((nDims.height - dims.height) / 2) },
            ];
            for (const pos of positions) {
                tryPlacement(pos.x, pos.y, orient);
            }
        }
    }

    const anchor = getRepairAnchor(connectedPlacedNeighbors, machineMap, original);
    for (let attempt = 0; attempt < 24; attempt++) {
        const orient = ORIENTATIONS[Math.floor(random() * ORIENTATIONS.length)];
        const x = Math.round(anchor.x + (random() * 10 - 5));
        const y = Math.round(anchor.y + (random() * 10 - 5));
        tryPlacement(x, y, orient);
    }
    tryPlacement(original.x, original.y, original.orientation);

    machine.x = original.x;
    machine.y = original.y;
    machine.orientation = original.orientation;
    if (!found) return null;
    return { x: bestX, y: bestY, orientation: bestOrientation };
}

function getRepairAnchor(
    neighborIds: Set<string>,
    machineMap: Map<string, Machine>,
    fallback: { x: number; y: number },
): { x: number; y: number } {
    if (neighborIds.size === 0) return fallback;
    let totalX = 0;
    let totalY = 0;
    let count = 0;
    for (const id of neighborIds) {
        const machine = machineMap.get(id);
        if (!machine) continue;
        totalX += machine.x;
        totalY += machine.y;
        count++;
    }
    if (count === 0) return fallback;
    return { x: totalX / count, y: totalY / count };
}

function estimateMachineConnectionCost(
    machineId: string,
    machines: Machine[],
    connections: Connection[],
): number {
    const machineMap = new Map(machines.map((m) => [m.id, m]));
    let totalDist = 0;
    for (const conn of connections) {
        if (conn.sourceMachineId !== machineId && conn.targetMachineId !== machineId) continue;
        const src = machineMap.get(conn.sourceMachineId);
        const tgt = machineMap.get(conn.targetMachineId);
        if (!src || !tgt) continue;
        const srcPort = getMachinePorts(src).outputs[conn.sourcePortIndex];
        const tgtPort = getMachinePorts(tgt).inputs[conn.targetPortIndex];
        if (!srcPort || !tgtPort) return Infinity;
        totalDist += estimateBeltLength(srcPort, tgtPort);
    }
    return totalDist;
}

function moveTowardNeighbor(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
    random: RandomFn,
    fixedMachineIds: Set<string>,
): void {
    const movableIndexes = getMovableMachineIndexes(machines, fixedMachineIds);
    if (movableIndexes.length === 0) return;
    const idx = movableIndexes[Math.floor(random() * movableIndexes.length)];
    const machine = machines[idx];

    const neighborCounts = new Map<string, number>();
    for (const conn of connections) {
        if (conn.sourceMachineId === machine.id) {
            neighborCounts.set(conn.targetMachineId, (neighborCounts.get(conn.targetMachineId) || 0) + 1);
        }
        if (conn.targetMachineId === machine.id) {
            neighborCounts.set(conn.sourceMachineId, (neighborCounts.get(conn.sourceMachineId) || 0) + 1);
        }
    }

    if (neighborCounts.size === 0) return;

    let bestNeighborId: string | null = null;
    let bestCount = 0;
    for (const [nid, count] of neighborCounts) {
        if (count > bestCount) {
            bestCount = count;
            bestNeighborId = nid;
        }
    }

    if (!bestNeighborId) return;
    const neighbor = machines.find((m) => m.id === bestNeighborId);
    if (!neighbor) return;

    const dx = Math.sign(neighbor.x - machine.x);
    const dy = Math.sign(neighbor.y - machine.y);
    const step = Math.floor(random() * 3) + 1;
    machine.x += dx * step;
    machine.y += dy * step;
    const dims = getOrientedDimensions(machine);
    machine.x = Math.max(0, Math.min(gridW - dims.width, machine.x));
    machine.y = Math.max(0, Math.min(gridH - dims.height, machine.y));
}

function moveToSource(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
    random: RandomFn,
    fixedMachineIds: Set<string>,
): void {
    const machineMap = new Map(machines.map((m) => [m.id, m]));
    const candidates = machines.filter((m) => {
        if (fixedMachineIds.has(m.id)) return false;
        return connections.some((conn) => conn.targetMachineId === m.id);
    });
    if (candidates.length === 0) return;

    const machine = candidates[Math.floor(random() * candidates.length)];
    let totalX = 0;
    let totalY = 0;
    let count = 0;
    for (const conn of connections) {
        if (conn.targetMachineId !== machine.id) continue;
        const source = machineMap.get(conn.sourceMachineId);
        if (!source) continue;
        totalX += source.x;
        totalY += source.y;
        count++;
    }
    if (count === 0) return;

    const targetX = totalX / count;
    const targetY = totalY / count;
    const dx = Math.sign(targetX - machine.x);
    const dy = Math.sign(targetY - machine.y);
    const majorStep = 1 + Math.floor(random() * 2);

    if (Math.abs(targetX - machine.x) >= Math.abs(targetY - machine.y)) {
        machine.x += dx * majorStep;
        if (random() < 0.6) machine.y += dy;
    } else {
        machine.y += dy * majorStep;
        if (random() < 0.6) machine.x += dx;
    }

    const dims = getOrientedDimensions(machine);
    machine.x = Math.max(0, Math.min(gridW - dims.width, machine.x));
    machine.y = Math.max(0, Math.min(gridH - dims.height, machine.y));
}

function tryDifferentPortAssignment(
    machines: Machine[],
    connections: Connection[],
    random: RandomFn,
    fixedMachineIds: Set<string>,
): void {
    const machineMap = new Map(machines.map((m) => [m.id, m]));
    const eligibleConnections = connections.filter((conn) => (
        !fixedMachineIds.has(conn.sourceMachineId) || !fixedMachineIds.has(conn.targetMachineId)
    ));
    if (eligibleConnections.length === 0) return;
    const conn = eligibleConnections[Math.floor(random() * eligibleConnections.length)];
    const src = machineMap.get(conn.sourceMachineId);
    const tgt = machineMap.get(conn.targetMachineId);
    if (!src || !tgt) return;

    const srcPortCount = getOutputPortCount(src);
    const tgtPortCount = getInputPortCount(tgt);
    if (srcPortCount <= 0 || tgtPortCount <= 0) return;

    const usedOutputPorts = new Set<number>();
    const usedInputPorts = new Set<number>();
    for (const other of connections) {
        if (other.id === conn.id) continue;
        if (other.sourceMachineId === src.id) usedOutputPorts.add(other.sourcePortIndex);
        if (other.targetMachineId === tgt.id) usedInputPorts.add(other.targetPortIndex);
    }

    const srcPorts = getMachinePorts(src).outputs;
    const tgtPorts = getMachinePorts(tgt).inputs;
    let bestSrcIdx = conn.sourcePortIndex;
    let bestTgtIdx = conn.targetPortIndex;
    let bestDist = Infinity;

    for (let si = 0; si < srcPortCount; si++) {
        if (usedOutputPorts.has(si)) continue;
        const srcPort = srcPorts[si];
        if (!srcPort) continue;
        for (let ti = 0; ti < tgtPortCount; ti++) {
            if (usedInputPorts.has(ti)) continue;
            const tgtPort = tgtPorts[ti];
            if (!tgtPort) continue;
            const dist = estimateBeltLength(srcPort, tgtPort);
            if (dist < bestDist) {
                bestDist = dist;
                bestSrcIdx = si;
                bestTgtIdx = ti;
            }
        }
    }

    if (bestDist < Infinity) {
        conn.sourcePortIndex = bestSrcIdx;
        conn.targetPortIndex = bestTgtIdx;
    }
}

/**
 * Teleport a random machine to the optimal port-facing side of its
 * most-connected neighbor. This is the SA equivalent of the greedy
 * port-facing placement — allows the SA to "rediscover" optimal
 * positions even if the initial placement was suboptimal.
 */
function portFacingJump(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
    random: RandomFn,
    fixedMachineIds: Set<string>,
): void {
    const movableIndexes = getMovableMachineIndexes(machines, fixedMachineIds);
    if (movableIndexes.length === 0) return;
    const idx = movableIndexes[Math.floor(random() * movableIndexes.length)];
    const machine = machines[idx];

    // Find most-connected neighbor
    const neighborCounts = new Map<string, number>();
    for (const conn of connections) {
        if (conn.sourceMachineId === machine.id) {
            neighborCounts.set(conn.targetMachineId, (neighborCounts.get(conn.targetMachineId) || 0) + 1);
        }
        if (conn.targetMachineId === machine.id) {
            neighborCounts.set(conn.sourceMachineId, (neighborCounts.get(conn.sourceMachineId) || 0) + 1);
        }
    }

    if (neighborCounts.size === 0) return;

    let bestNeighborId: string | null = null;
    let bestCount = 0;
    for (const [nid, count] of neighborCounts) {
        if (count > bestCount) {
            bestCount = count;
            bestNeighborId = nid;
        }
    }
    if (!bestNeighborId) return;

    const neighbor = machines.find((m) => m.id === bestNeighborId);
    if (!neighbor) return;

    const nDims = getOrientedDimensions(neighbor);
    const machineMap = new Map(machines.map((m) => [m.id, m]));
    const placed = new Set(machines.filter((m) => m.id !== machine.id).map((m) => m.id));

    let bestCost = Infinity;
    let bestX = machine.x;
    let bestY = machine.y;
    let bestOrient = machine.orientation;

    for (const orient of ORIENTATIONS) {
        machine.orientation = orient;
        const dims = getOrientedDimensions(machine);

        // Try key positions on each side
        const positions = [
            { x: neighbor.x, y: neighbor.y + nDims.height + 1 },  // below, left-aligned
            { x: neighbor.x + Math.floor((nDims.width - dims.width) / 2), y: neighbor.y + nDims.height + 1 },  // below, centered
            { x: neighbor.x, y: neighbor.y - dims.height - 1 },   // above
            { x: neighbor.x + Math.floor((nDims.width - dims.width) / 2), y: neighbor.y - dims.height - 1 },
            { x: neighbor.x + nDims.width + 1, y: neighbor.y },   // right
            { x: neighbor.x + nDims.width + 1, y: neighbor.y + Math.floor((nDims.height - dims.height) / 2) },
            { x: neighbor.x - dims.width - 1, y: neighbor.y },    // left
            { x: neighbor.x - dims.width - 1, y: neighbor.y + Math.floor((nDims.height - dims.height) / 2) },
        ];

        for (const pos of positions) {
            machine.x = pos.x;
            machine.y = pos.y;
            if (!isValidPlacement(machine, machines, placed, gridW, gridH)) continue;

            let totalDist = 0;
            for (const conn of connections) {
                if (conn.sourceMachineId !== machine.id && conn.targetMachineId !== machine.id) continue;
                const src = machineMap.get(conn.sourceMachineId);
                const tgt = machineMap.get(conn.targetMachineId);
                if (!src || !tgt) continue;
                const sp = getMachinePorts(src).outputs[conn.sourcePortIndex];
                const tp = getMachinePorts(tgt).inputs[conn.targetPortIndex];
                if (sp && tp) totalDist += estimateBeltLength(sp, tp);
            }

            if (totalDist < bestCost) {
                bestCost = totalDist;
                bestX = pos.x;
                bestY = pos.y;
                bestOrient = orient;
            }
        }
    }

    machine.x = bestX;
    machine.y = bestY;
    machine.orientation = bestOrient;
}

// ─────────────────────────────────────────────────────────
// PHASE 3: PORT ASSIGNMENT OPTIMIZATION
// ─────────────────────────────────────────────────────────

/**
 * For each connection, try all valid port assignment pairs
 * and pick the one that gives the shortest Manhattan distance.
 */
function optimizePortAssignments(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
): Machine[] {
    const result = machines.map((m) => ({ ...m }));
    const machineMap = new Map(result.map((m) => [m.id, m]));
    const originalConns = connections.map((c) => ({ ...c }));
    const optimizedConns = connections.map((c) => ({ ...c }));

    const usedOutputPorts = new Map<string, Set<number>>();
    const usedInputPorts = new Map<string, Set<number>>();

    // Sort connections by estimated length (longest first = most benefit)
    const withEstimates = optimizedConns.map((conn) => {
        const src = machineMap.get(conn.sourceMachineId);
        const tgt = machineMap.get(conn.targetMachineId);
        if (!src || !tgt) return { conn, est: Infinity };
        const srcPorts = getMachinePorts(src);
        const tgtPorts = getMachinePorts(tgt);
        const srcPort = srcPorts.outputs[conn.sourcePortIndex];
        const tgtPort = tgtPorts.inputs[conn.targetPortIndex];
        const est = srcPort && tgtPort ? estimateBeltLength(srcPort, tgtPort) : Infinity;
        return { conn, est };
    });
    withEstimates.sort((a, b) => b.est - a.est);

    for (const { conn } of withEstimates) {
        const src = machineMap.get(conn.sourceMachineId);
        const tgt = machineMap.get(conn.targetMachineId);
        if (!src || !tgt) continue;
        const srcPorts = getMachinePorts(src);
        const tgtPorts = getMachinePorts(tgt);
        const srcPortCount = getOutputPortCount(src);
        const tgtPortCount = getInputPortCount(tgt);
        if (srcPortCount <= 0 || tgtPortCount <= 0) continue;

        const srcUsed = usedOutputPorts.get(src.id) || new Set();
        const tgtUsed = usedInputPorts.get(tgt.id) || new Set();

        let bestSrcIdx = conn.sourcePortIndex;
        let bestTgtIdx = conn.targetPortIndex;
        let bestDist = Infinity;

        for (let si = 0; si < srcPortCount; si++) {
            if (srcUsed.has(si)) continue;
            for (let ti = 0; ti < tgtPortCount; ti++) {
                if (tgtUsed.has(ti)) continue;
                const sp = srcPorts.outputs[si];
                const tp = tgtPorts.inputs[ti];
                if (!sp || !tp) continue;
                const dist = estimateBeltLength(sp, tp);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestSrcIdx = si;
                    bestTgtIdx = ti;
                }
            }
        }

        conn.sourcePortIndex = bestSrcIdx;
        conn.targetPortIndex = bestTgtIdx;
        srcUsed.add(bestSrcIdx);
        tgtUsed.add(bestTgtIdx);
        usedOutputPorts.set(src.id, srcUsed);
        usedInputPorts.set(tgt.id, tgtUsed);
    }

    const chosen = chooseConnectionAssignment(result, originalConns, optimizedConns, gridW, gridH);

    connections.length = 0;
    connections.push(...chosen.map((c) => ({ ...c })));
    return result;
}

function chooseConnectionAssignment(
    machines: Machine[],
    originalConns: Connection[],
    optimizedConns: Connection[],
    gridW: number,
    gridH: number,
): Connection[] {
    const originalRouted = buildAndScore(machines, originalConns, gridW, gridH);
    const optimizedRouted = buildAndScore(machines, optimizedConns, gridW, gridH);

    if (originalRouted && optimizedRouted) {
        return optimizedRouted.score.totalScore <= originalRouted.score.totalScore ? optimizedConns : originalConns;
    }
    if (optimizedRouted) return optimizedConns;
    if (originalRouted) return originalConns;

    const machineMap = new Map(machines.map((m) => [m.id, m]));
    const originalFast = computeFastScore(machines, originalConns, machineMap);
    const optimizedFast = computeFastScore(machines, optimizedConns, machineMap);
    return optimizedFast <= originalFast ? optimizedConns : originalConns;
}

// ─────────────────────────────────────────────────────────
// PHASE 4: COMPACTION
// ─────────────────────────────────────────────────────────

/**
 * Slide all machines toward top-left to minimize bounding box.
 * Also tries compacting toward center of mass for better belt routing.
 */
function compactLayout(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
    fixedMachines: Map<string, Machine> = new Map(),
): Machine[] {
    const baseline = machines.map((m) => ({ ...m }));
    enforceFixedMachines(baseline, fixedMachines);
    const baselineScore = scorePlacement(baseline, connections, gridW, gridH, false);
    const result = baseline.map((m) => ({ ...m }));
    enforceFixedMachines(result, fixedMachines);
    const allIds = new Set(result.map((m) => m.id));

    // Shift bounding box to (1,1)
    let minX = Infinity, minY = Infinity;
    for (const m of result) {
        if (fixedMachines.has(m.id)) continue;
        minX = Math.min(minX, m.x);
        minY = Math.min(minY, m.y);
    }
    if (Number.isFinite(minX) && Number.isFinite(minY)) {
        const shiftX = minX - 1;
        const shiftY = minY - 1;
        for (const m of result) {
            if (fixedMachines.has(m.id)) continue;
            m.x -= shiftX;
            m.y -= shiftY;
        }
    }

    // Repeatedly slide each machine left and up
    let improved = true;
    let passes = 0;
    while (improved && passes < 30) {
        improved = false;
        passes++;
        result.sort((a, b) => (a.x + a.y) - (b.x + b.y));

        for (const machine of result) {
            if (fixedMachines.has(machine.id)) continue;
            while (machine.x > 0) {
                machine.x--;
                if (!isValidPlacement(machine, result, allIds, gridW, gridH)) {
                    machine.x++;
                    break;
                }
                improved = true;
            }
            while (machine.y > 0) {
                machine.y--;
                if (!isValidPlacement(machine, result, allIds, gridW, gridH)) {
                    machine.y++;
                    break;
                }
                improved = true;
            }
        }
        enforceFixedMachines(result, fixedMachines);
    }

    const compactedScore = scorePlacement(result, connections, gridW, gridH, false);
    return compactedScore <= baselineScore ? result : baseline;
}

// ─────────────────────────────────────────────────────────
// GRID BUILDER
// ─────────────────────────────────────────────────────────

function buildGrid(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
): GridState | null {
    const grid = createGrid(gridW, gridH);

    for (const m of machines) {
        if (!placeMachine(grid, m)) return null;
    }

    for (const conn of connections) {
        grid.connections.set(conn.id, conn);
    }

    for (const conn of connections) {
        const src = grid.machines.get(conn.sourceMachineId);
        const tgt = grid.machines.get(conn.targetMachineId);
        if (!src || !tgt) return null;

        const srcPorts = getMachinePorts(src);
        const tgtPorts = getMachinePorts(tgt);
        const srcPort = srcPorts.outputs[conn.sourcePortIndex];
        const tgtPort = tgtPorts.inputs[conn.targetPortIndex];
        if (!srcPort || !tgtPort) return null;

        const path = findBeltPath(grid, srcPort, tgtPort, conn.id);
        if (!path) return null;
        applyBeltPath(grid, path);
    }

    return grid;
}

function buildAndScore(
    machines: Machine[],
    connections: Connection[],
    gridW: number,
    gridH: number,
): { grid: GridState; score: ScoreBreakdown } | null {
    const grid = buildGrid(machines, connections, gridW, gridH);
    if (!grid) return null;
    return { grid, score: evaluateGrid(grid) };
}
