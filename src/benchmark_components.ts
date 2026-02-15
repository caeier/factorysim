import { evaluateGrid, type ScoreBreakdown } from './scoring';
import { runOptimizer } from './optimizer';
import {
    type BenchmarkScenario,
    createAdversarialScenarios,
    createGridFromScenario,
    createRandomDataset,
} from './benchmark_scenarios';

const DEEP_BENCHMARK_CONFIG = {
    mode: 'deep' as const,
    timeBudgetMs: 2400,
    useExplorationSeeds: true,
    phase1Restarts: 2,
    phase2Attempts: 2,
    localPolishPasses: 2,
    elitePoolSize: 12,
    eliteDiversityHash: true,
    eliteMinDistance: 0.75,
    clusterMoveMinSize: 2,
    clusterMoveMaxSize: 5,
    adaptiveOps: true,
    adaptiveWindow: 100,
    adaptiveWarmupIterations: 100,
    adaptiveMaxOperatorProb: 0.45,
    adaptiveStagnationResetWindow: 160,
    adaptiveFlattenFactor: 0.55,
    largeMoveRate: 0.02,
    largeMoveRateEarly: 0.03,
    largeMoveRateLate: 0.005,
    largeMoveCooldownAfterImprove: 70,
    criticalNetRate: 0.005,
    repairBeamWidth: 1,
};

interface ComponentDelta {
    scenario: string;
    start: ScoreBreakdown;
    final: ScoreBreakdown;
    runtimeMs: number;
    iterations: number;
}

function stableSeedFromName(name: string): number {
    let hash = 2166136261;
    for (let i = 0; i < name.length; i++) {
        hash ^= name.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
}

function fmt(n: number): string {
    return n.toFixed(1);
}

function fmtDelta(v: number): string {
    return (v >= 0 ? '+' : '') + v.toFixed(1);
}

async function runScenario(scenario: BenchmarkScenario): Promise<ComponentDelta> {
    const startGrid = createGridFromScenario(scenario);
    const start = evaluateGrid(startGrid);
    const seed = stableSeedFromName(scenario.name);

    const startedAt = performance.now();
    const result = await runOptimizer(startGrid, { ...DEEP_BENCHMARK_CONFIG, seed });
    const runtimeMs = performance.now() - startedAt;

    return {
        scenario: scenario.name,
        start,
        final: result.score,
        runtimeMs,
        iterations: result.iterations,
    };
}

function printRow(result: ComponentDelta): void {
    const dBelts = result.start.totalBelts - result.final.totalBelts;
    const dArea = result.start.boundingBoxArea - result.final.boundingBoxArea;
    const dCorners = result.start.cornerCount - result.final.cornerCount;
    const dScore = result.start.totalScore - result.final.totalScore;

    console.log(
        [
            `- ${result.scenario}`,
            `belts ${fmt(result.start.totalBelts)} -> ${fmt(result.final.totalBelts)} (${fmtDelta(dBelts)})`,
            `area ${fmt(result.start.boundingBoxArea)} -> ${fmt(result.final.boundingBoxArea)} (${fmtDelta(dArea)})`,
            `corners ${fmt(result.start.cornerCount)} -> ${fmt(result.final.cornerCount)} (${fmtDelta(dCorners)})`,
            `score ${fmt(result.start.totalScore)} -> ${fmt(result.final.totalScore)} (${fmtDelta(dScore)})`,
            `time=${result.runtimeMs.toFixed(0)}ms`,
            `iter=${result.iterations}`,
        ].join(' | '),
    );
}

function printAggregate(label: string, results: ComponentDelta[]): void {
    let beltsGain = 0;
    let areaGain = 0;
    let cornerGain = 0;
    let scoreGain = 0;
    for (const result of results) {
        beltsGain += result.start.totalBelts - result.final.totalBelts;
        areaGain += result.start.boundingBoxArea - result.final.boundingBoxArea;
        cornerGain += result.start.cornerCount - result.final.cornerCount;
        scoreGain += result.start.totalScore - result.final.totalScore;
    }
    const count = Math.max(1, results.length);
    console.log(`\n--- ${label} component summary ---`);
    console.log(`avg belts delta: ${(beltsGain / count).toFixed(2)}`);
    console.log(`avg area delta: ${(areaGain / count).toFixed(2)}`);
    console.log(`avg corners delta: ${(cornerGain / count).toFixed(2)}`);
    console.log(`avg total-score delta: ${(scoreGain / count).toFixed(2)}`);
}

async function runGroup(label: string, scenarios: BenchmarkScenario[]): Promise<ComponentDelta[]> {
    console.log(`\n=== ${label.toUpperCase()} COMPONENT DELTAS ===`);
    const results: ComponentDelta[] = [];
    for (const scenario of scenarios) {
        const result = await runScenario(scenario);
        results.push(result);
        printRow(result);
    }
    printAggregate(label, results);
    return results;
}

async function run(): Promise<void> {
    const adversarial = createAdversarialScenarios();
    const tuning = createRandomDataset('tuning');
    const holdout = createRandomDataset('holdout');

    await runGroup('adversarial curated', adversarial);
    await runGroup('random tuning', tuning);
    await runGroup('random holdout', holdout);
}

run().catch((error) => {
    console.error(error);
});
