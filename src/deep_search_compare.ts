import { evaluateGrid } from './scoring';
import { runOptimizer } from './optimizer';
import {
    createAdversarialScenarios,
    createGridFromScenario,
    createRandomDataset,
    summarize,
} from './benchmark_scenarios';

interface RunStats {
    scenario: string;
    startScore: number;
    finalScore: number;
    improvement: number;
    runtimeMs: number;
    improvePerSec: number;
    iterations: number;
    regressed: boolean;
}

const BASELINE_CONFIG = {
    mode: 'deep' as const,
    timeBudgetMs: 2600,
    phase1Restarts: 1,
    phase2Attempts: 1,
    localPolishPasses: 1,
    elitePoolSize: 1,
    eliteDiversityHash: false,
    eliteMinDistance: 0,
    largeMoveRate: 0,
    clusterMoveMinSize: 2,
    clusterMoveMaxSize: 4,
    adaptiveOps: false,
    adaptiveWindow: 100,
};

const UPGRADED_CONFIG = {
    mode: 'deep' as const,
    timeBudgetMs: 2600,
    useExplorationSeeds: true,
    phase1Restarts: 2,
    phase2Attempts: 2,
    localPolishPasses: 2,
    elitePoolSize: 12,
    eliteDiversityHash: true,
    eliteMinDistance: 0.75,
    largeMoveRate: 0.02,
    clusterMoveMinSize: 2,
    clusterMoveMaxSize: 5,
    adaptiveOps: true,
    adaptiveWindow: 100,
    adaptiveWarmupIterations: 100,
    adaptiveMaxOperatorProb: 0.45,
    adaptiveStagnationResetWindow: 160,
    adaptiveFlattenFactor: 0.55,
    largeMoveRateEarly: 0.03,
    largeMoveRateLate: 0.005,
    largeMoveCooldownAfterImprove: 70,
    criticalNetRate: 0.005,
    repairBeamWidth: 1,
};

async function runConfig(label: string, config: Record<string, unknown>): Promise<RunStats[]> {
    const scenarios = [...createAdversarialScenarios(), ...createRandomDataset('holdout')];
    const results: RunStats[] = [];

    console.log(`\n=== ${label} ===`);
    for (const scenario of scenarios) {
        const grid = createGridFromScenario(scenario);
        const startScore = evaluateGrid(grid).totalScore;
        const seed = stableSeedFromName(scenario.name);
        const startedAt = performance.now();
        const outcome = await runOptimizer(grid, { ...config, seed });
        const runtimeMs = performance.now() - startedAt;
        const finalScore = outcome.score.totalScore;
        const improvement = startScore - finalScore;
        const improvePerSec = runtimeMs > 0 ? improvement / (runtimeMs / 1000) : 0;
        const regressed = finalScore > startScore + 1e-6;

        results.push({
            scenario: scenario.name,
            startScore,
            finalScore,
            improvement,
            runtimeMs,
            improvePerSec,
            iterations: outcome.iterations,
            regressed,
        });

        const regressedFlag = regressed ? ' [REGRESSION]' : '';
        console.log(
            [
                `- ${scenario.name}`,
                `start=${startScore.toFixed(1)}`,
                `final=${finalScore.toFixed(1)}`,
                `improve=${improvement.toFixed(2)}`,
                `improve/s=${improvePerSec.toFixed(2)}`,
                `time=${runtimeMs.toFixed(0)}ms`,
                `iter=${outcome.iterations}`,
            ].join(' | ') + regressedFlag,
        );
    }

    return results;
}

function printAggregate(label: string, runs: RunStats[]): void {
    const improve = summarize(runs.map((r) => r.improvement));
    const improvePerSec = summarize(runs.map((r) => r.improvePerSec));
    const finals = summarize(runs.map((r) => r.finalScore));
    const regressions = runs.filter((r) => r.regressed).length;

    console.log(`\n--- ${label} aggregate ---`);
    console.log(`improvement: mean=${improve.mean.toFixed(2)} p50=${improve.p50.toFixed(2)} p90=${improve.p90.toFixed(2)}`);
    console.log(`improve/sec: mean=${improvePerSec.mean.toFixed(2)} p50=${improvePerSec.p50.toFixed(2)} p90=${improvePerSec.p90.toFixed(2)}`);
    console.log(`final score: mean=${finals.mean.toFixed(2)} p50=${finals.p50.toFixed(2)} p90=${finals.p90.toFixed(2)}`);
    console.log(`no-regression violations: ${regressions}/${runs.length}`);
}

function printABSummary(baseline: RunStats[], upgraded: RunStats[]): void {
    let improvementDeltaTotal = 0;
    let improvePerSecDeltaTotal = 0;
    let betterCount = 0;

    for (let i = 0; i < baseline.length; i++) {
        const b = baseline[i];
        const u = upgraded[i];
        const improvementDelta = u.improvement - b.improvement;
        const improvePerSecDelta = u.improvePerSec - b.improvePerSec;
        improvementDeltaTotal += improvementDelta;
        improvePerSecDeltaTotal += improvePerSecDelta;
        if (u.finalScore < b.finalScore - 1e-6) betterCount++;
    }

    const count = Math.max(1, baseline.length);
    console.log('\n=== A/B summary (upgraded - baseline) ===');
    console.log(`avg improvement delta: ${(improvementDeltaTotal / count).toFixed(2)}`);
    console.log(`avg improve/sec delta: ${(improvePerSecDeltaTotal / count).toFixed(2)}`);
    console.log(`upgraded better final score on ${betterCount}/${count} scenarios`);
}

function stableSeedFromName(name: string): number {
    let hash = 2166136261;
    for (let i = 0; i < name.length; i++) {
        hash ^= name.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
}

async function run(): Promise<void> {
    const baseline = await runConfig('baseline deep', BASELINE_CONFIG);
    const upgraded = await runConfig('upgraded deep', UPGRADED_CONFIG);

    printAggregate('baseline deep', baseline);
    printAggregate('upgraded deep', upgraded);
    printABSummary(baseline, upgraded);
}

run().catch((error) => {
    console.error(error);
});
