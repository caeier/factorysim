import { evaluateGrid } from './scoring';
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
    startScore: number;
    finalScore: number;
    improvement: number;
    runtimeMs: number;
    iterations: number;
    regressed: boolean;
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
    const startScore = evaluateGrid(startGrid).totalScore;
    const seed = stableSeedFromName(scenario.name);

    const startedAt = performance.now();
    const result = await runOptimizer(startGrid, { ...DEEP_BENCHMARK_CONFIG, seed });
    const runtimeMs = performance.now() - startedAt;
    const finalScore = result.score.totalScore;

    return {
        name: scenario.name,
        startScore,
        finalScore,
        improvement: startScore - finalScore,
        runtimeMs,
        iterations: result.iterations,
        regressed: finalScore > startScore + 1e-6,
    };
}

async function runGroup(label: string, scenarios: BenchmarkScenario[]): Promise<ScenarioResult[]> {
    console.log(`\n=== ${label.toUpperCase()} ===`);
    const results: ScenarioResult[] = [];
    for (const scenario of scenarios) {
        const outcome = await runScenario(scenario);
        results.push(outcome);
        const regressionFlag = outcome.regressed ? '  [REGRESSION]' : '';
        console.log(
            [
                `- ${outcome.name}`,
                `start=${outcome.startScore.toFixed(1)}`,
                `final=${outcome.finalScore.toFixed(1)}`,
                `improve=${outcome.improvement.toFixed(1)}`,
                `time=${outcome.runtimeMs.toFixed(0)}ms`,
                `iter=${outcome.iterations}`,
            ].join(' | ') + regressionFlag,
        );
    }
    return results;
}

function printSummary(label: string, results: ScenarioResult[]): void {
    const finalScores = results.map((r) => r.finalScore);
    const runtimes = results.map((r) => r.runtimeMs);
    const iterations = results.map((r) => r.iterations);
    const improvements = results.map((r) => r.improvement);
    const scoreStats = summarize(finalScores);
    const runtimeStats = summarize(runtimes);
    const iterStats = summarize(iterations);
    const improveStats = summarize(improvements);
    const regressionCount = results.filter((r) => r.regressed).length;

    console.log(`\n--- ${label} summary ---`);
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
        `improvement: mean=${improveStats.mean.toFixed(2)} p50=${improveStats.p50.toFixed(2)} p90=${improveStats.p90.toFixed(2)}`,
    );
    console.log(`no-regression violations: ${regressionCount}/${results.length}`);
}

function stableSeedFromName(name: string): number {
    let hash = 2166136261;
    for (let i = 0; i < name.length; i++) {
        hash ^= name.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
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
