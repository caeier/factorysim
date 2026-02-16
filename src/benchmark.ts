import { compareScoreBreakdownLexicographic, evaluateGrid, type ScoreBreakdown } from './scoring';
import { runOptimizer } from './optimizer';
import {
    type BenchmarkScenario,
    createAdversarialScenarios,
    createGridFromScenario,
    createRandomDataset,
    summarize,
} from './benchmark_scenarios';

interface ScenarioResult {
    name: string;
    start: ScoreBreakdown;
    final: ScoreBreakdown;
    totalScoreDelta: number;
    runtimeMs: number;
    iterations: number;
    priorityComparison: number;
}

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

async function runScenario(scenario: BenchmarkScenario): Promise<ScenarioResult> {
    const startGrid = createGridFromScenario(scenario);
    const start = evaluateGrid(startGrid);
    const seed = stableSeedFromName(scenario.name);

    const startedAt = performance.now();
    const result = await runOptimizer(startGrid, { ...DEEP_BENCHMARK_CONFIG, seed });
    const runtimeMs = performance.now() - startedAt;
    const final = result.score;

    return {
        name: scenario.name,
        start,
        final,
        totalScoreDelta: start.totalScore - final.totalScore,
        runtimeMs,
        iterations: result.iterations,
        priorityComparison: compareScoreBreakdownLexicographic(final, start),
    };
}

async function runGroup(label: string, scenarios: BenchmarkScenario[]): Promise<ScenarioResult[]> {
    console.log(`\n=== ${label.toUpperCase()} ===`);
    const results: ScenarioResult[] = [];
    for (const scenario of scenarios) {
        const outcome = await runScenario(scenario);
        results.push(outcome);
        const priorityStatus = formatPriorityStatus(outcome.priorityComparison);
        const regressionFlag = outcome.priorityComparison > 1e-6 ? '  [PRIORITY REGRESSION]' : '';
        console.log(
            [
                `- ${outcome.name}`,
                `belts ${outcome.start.totalBelts.toFixed(1)} -> ${outcome.final.totalBelts.toFixed(1)}`,
                `area ${outcome.start.boundingBoxArea.toFixed(1)} -> ${outcome.final.boundingBoxArea.toFixed(1)}`,
                `corners ${outcome.start.cornerCount.toFixed(1)} -> ${outcome.final.cornerCount.toFixed(1)}`,
                `score ${outcome.start.totalScore.toFixed(1)} -> ${outcome.final.totalScore.toFixed(1)} (delta=${outcome.totalScoreDelta.toFixed(1)})`,
                `priority=${priorityStatus}`,
                `time=${outcome.runtimeMs.toFixed(0)}ms`,
                `iter=${outcome.iterations}`,
            ].join(' | ') + regressionFlag,
        );
    }
    return results;
}

function printSummary(label: string, results: ScenarioResult[]): void {
    const finalBelts = results.map((r) => r.final.totalBelts);
    const finalAreas = results.map((r) => r.final.boundingBoxArea);
    const finalCorners = results.map((r) => r.final.cornerCount);
    const finalScores = results.map((r) => r.final.totalScore);
    const runtimes = results.map((r) => r.runtimeMs);
    const iterations = results.map((r) => r.iterations);
    const scoreDeltas = results.map((r) => r.totalScoreDelta);
    const beltsStats = summarize(finalBelts);
    const areaStats = summarize(finalAreas);
    const cornerStats = summarize(finalCorners);
    const scoreStats = summarize(finalScores);
    const runtimeStats = summarize(runtimes);
    const iterStats = summarize(iterations);
    const deltaStats = summarize(scoreDeltas);
    const improvedCount = results.filter((r) => r.priorityComparison < -1e-6).length;
    const tiedCount = results.filter((r) => Math.abs(r.priorityComparison) <= 1e-6).length;
    const regressionCount = results.filter((r) => r.priorityComparison > 1e-6).length;

    console.log(`\n--- ${label} summary ---`);
    console.log(
        `final belts: mean=${beltsStats.mean.toFixed(2)} p50=${beltsStats.p50.toFixed(2)} p90=${beltsStats.p90.toFixed(2)}`,
    );
    console.log(
        `final area:  mean=${areaStats.mean.toFixed(2)} p50=${areaStats.p50.toFixed(2)} p90=${areaStats.p90.toFixed(2)}`,
    );
    console.log(
        `final corner: mean=${cornerStats.mean.toFixed(2)} p50=${cornerStats.p50.toFixed(2)} p90=${cornerStats.p90.toFixed(2)}`,
    );
    console.log(
        `final score: mean=${scoreStats.mean.toFixed(2)} p50=${scoreStats.p50.toFixed(2)} p90=${scoreStats.p90.toFixed(2)}`,
    );
    console.log(
        `runtime ms:  mean=${runtimeStats.mean.toFixed(0)} p50=${runtimeStats.p50.toFixed(0)} p90=${runtimeStats.p90.toFixed(0)}`,
    );
    console.log(
        `iterations:  mean=${iterStats.mean.toFixed(0)} p50=${iterStats.p50.toFixed(0)} p90=${iterStats.p90.toFixed(0)}`,
    );
    console.log(
        `score delta: mean=${deltaStats.mean.toFixed(2)} p50=${deltaStats.p50.toFixed(2)} p90=${deltaStats.p90.toFixed(2)}`,
    );
    console.log(`priority result counts: improved=${improvedCount} tied=${tiedCount} regressed=${regressionCount}`);
}

function stableSeedFromName(name: string): number {
    let hash = 2166136261;
    for (let i = 0; i < name.length; i++) {
        hash ^= name.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
}

function formatPriorityStatus(comparison: number): string {
    if (comparison < -1e-6) return 'improved';
    if (comparison > 1e-6) return 'regressed';
    return 'tied';
}

async function runBenchmark(): Promise<void> {
    const adversarial = createAdversarialScenarios();
    const tuning = createRandomDataset('tuning');
    const holdout = createRandomDataset('holdout');

    const adversarialResults = await runGroup('adversarial curated', adversarial);
    const tuningResults = await runGroup('random tuning', tuning);
    const holdoutResults = await runGroup('random holdout', holdout);

    printSummary('adversarial curated', adversarialResults);
    printSummary('random tuning', tuningResults);
    printSummary('random holdout', holdoutResults);
}

runBenchmark().catch((error) => {
    console.error(error);
});
