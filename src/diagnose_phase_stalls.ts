import { runOptimizer, type OptimizerPhaseDiagnostics } from './optimizer';
import { compareScoreBreakdownLexicographic, evaluateGrid, type ScoreBreakdown } from './scoring';
import { createAdversarialScenarios, createGridFromScenario } from './benchmark_scenarios';

const DIAG_CONFIG = {
    mode: 'deep' as const,
    diagnostics: true,
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

function stableSeedFromName(name: string): number {
    let hash = 2166136261;
    for (let i = 0; i < name.length; i++) {
        hash ^= name.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
}

function formatBreakdown(score: ScoreBreakdown): string {
    return [
        `belts=${score.totalBelts.toFixed(1)}`,
        `area=${score.boundingBoxArea.toFixed(1)}`,
        `corners=${score.cornerCount.toFixed(1)}`,
        `score=${score.totalScore.toFixed(1)}`,
    ].join(' ');
}

function printPhaseLines(lines: OptimizerPhaseDiagnostics[]): void {
    for (const phase of lines) {
        console.log(
            [
                `  ${phase.phase.padEnd(7)}`,
                `${formatBreakdown(phase.startScore)} -> ${formatBreakdown(phase.endScore)}`,
                `time=${phase.durationMs.toFixed(0)}ms`,
                `iter_delta=${phase.iterationDelta}`,
            ].join(' | '),
        );
    }
}

async function run(): Promise<void> {
    const scenarios = createAdversarialScenarios();
    console.log(`running phase diagnostics for ${scenarios.length} adversarial scenarios`);

    for (const scenario of scenarios) {
        const grid = createGridFromScenario(scenario);
        const start = evaluateGrid(grid);
        const seed = stableSeedFromName(scenario.name);
        const startedAt = performance.now();
        const result = await runOptimizer(grid, { ...DIAG_CONFIG, seed });
        const runtimeMs = performance.now() - startedAt;
        const final = result.score;
        const priorityComparison = compareScoreBreakdownLexicographic(final, start);
        const priorityStatus = priorityComparison < -1e-6 ? 'improved' : priorityComparison > 1e-6 ? 'regressed' : 'tied';
        const diagnostics = result.diagnostics;

        console.log(`\n=== ${scenario.name} ===`);
        console.log(
            [
                `start: ${formatBreakdown(start)}`,
                `final: ${formatBreakdown(final)}`,
                `priority=${priorityStatus}`,
                `runtime=${runtimeMs.toFixed(0)}ms`,
                `iterations=${result.iterations}`,
                `first_strict_improvement_phase=${diagnostics?.firstStrictImprovementPhase ?? 'none'}`,
            ].join(' | '),
        );

        if (diagnostics?.phaseDiagnostics.length) {
            printPhaseLines(diagnostics.phaseDiagnostics);
        } else {
            console.log('  no phase diagnostics collected');
        }
    }
}

run().catch((error) => {
    console.error(error);
});
