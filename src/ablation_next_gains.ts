import { compareScoreBreakdownLexicographic, evaluateGrid, type ScoreBreakdown } from './scoring';
import { runOptimizer } from './optimizer';
import {
    createAdversarialScenarios,
    createGridFromScenario,
    createRandomDataset,
    summarize,
} from './benchmark_scenarios';

interface ScenarioRun {
    scenario: string;
    start: ScoreBreakdown;
    final: ScoreBreakdown;
    startScore: number;
    finalScore: number;
    improvement: number;
    runtimeMs: number;
    iterations: number;
    priorityComparison: number;
    regressed: boolean;
}

interface NamedConfig {
    name: string;
    config: Record<string, unknown>;
}

const SCENARIOS = [...createAdversarialScenarios(), ...createRandomDataset('holdout')];

const CONFIGS: NamedConfig[] = [
    {
        name: 'baseline',
        config: {
            mode: 'deep',
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
            repairBeamWidth: 1,
            criticalNetRate: 0,
        },
    },
    {
        name: 'archive_only',
        config: {
            mode: 'deep',
            timeBudgetMs: 2600,
            phase1Restarts: 2,
            phase2Attempts: 2,
            localPolishPasses: 2,
            elitePoolSize: 12,
            eliteDiversityHash: true,
            eliteMinDistance: 0.75,
            largeMoveRate: 0,
            clusterMoveMinSize: 2,
            clusterMoveMaxSize: 5,
            adaptiveOps: false,
            adaptiveWindow: 100,
            repairBeamWidth: 1,
            criticalNetRate: 0,
        },
    },
    {
        name: 'archive+adaptive_fixed',
        config: {
            mode: 'deep',
            timeBudgetMs: 2600,
            phase1Restarts: 2,
            phase2Attempts: 2,
            localPolishPasses: 2,
            elitePoolSize: 12,
            eliteDiversityHash: true,
            eliteMinDistance: 0.75,
            largeMoveRate: 0,
            clusterMoveMinSize: 2,
            clusterMoveMaxSize: 5,
            adaptiveOps: true,
            adaptiveWindow: 100,
            adaptiveWarmupIterations: 100,
            adaptiveMaxOperatorProb: 0.45,
            adaptiveStagnationResetWindow: 160,
            adaptiveFlattenFactor: 0.55,
            repairBeamWidth: 1,
            criticalNetRate: 0,
        },
    },
    {
        name: 'combined_fixed',
        config: {
            mode: 'deep',
            timeBudgetMs: 2600,
            phase1Restarts: 2,
            phase2Attempts: 2,
            localPolishPasses: 2,
            elitePoolSize: 12,
            eliteDiversityHash: true,
            eliteMinDistance: 0.75,
            largeMoveRate: 0.02,
            largeMoveRateEarly: 0.03,
            largeMoveRateLate: 0.005,
            largeMoveCooldownAfterImprove: 70,
            clusterMoveMinSize: 2,
            clusterMoveMaxSize: 5,
            adaptiveOps: true,
            adaptiveWindow: 100,
            adaptiveWarmupIterations: 100,
            adaptiveMaxOperatorProb: 0.45,
            adaptiveStagnationResetWindow: 160,
            adaptiveFlattenFactor: 0.55,
            criticalNetRate: 0.005,
            repairBeamWidth: 1,
        },
    },
];

async function runConfig(named: NamedConfig): Promise<ScenarioRun[]> {
    const results: ScenarioRun[] = [];
    console.log(`\n=== ${named.name} ===`);

    for (const scenario of SCENARIOS) {
        const grid = createGridFromScenario(scenario);
        const start = evaluateGrid(grid);
        const startScore = start.totalScore;
        const seed = stableSeedFromName(scenario.name);
        const startedAt = performance.now();
        const result = await runOptimizer(grid, { ...named.config, seed });
        const runtimeMs = performance.now() - startedAt;
        const final = result.score;
        const finalScore = final.totalScore;
        const improvement = startScore - finalScore;
        const priorityComparison = compareScoreBreakdownLexicographic(final, start);
        const regressed = priorityComparison > 1e-6;

        results.push({
            scenario: scenario.name,
            start,
            final,
            startScore,
            finalScore,
            improvement,
            runtimeMs,
            iterations: result.iterations,
            priorityComparison,
            regressed,
        });

        const regressedFlag = regressed ? ' [REGRESSION]' : '';
        console.log(
            [
                `- ${scenario.name}`,
                `start=${startScore.toFixed(1)}`,
                `final=${finalScore.toFixed(1)}`,
                `improve=${improvement.toFixed(1)}`,
                `priority=${priorityComparison < -1e-6 ? 'improved' : priorityComparison > 1e-6 ? 'regressed' : 'tied'}`,
                `time=${runtimeMs.toFixed(0)}ms`,
                `iter=${result.iterations}`,
            ].join(' | ') + regressedFlag,
        );
    }

    return results;
}

function printAggregate(label: string, runs: ScenarioRun[]): void {
    const improveStats = summarize(runs.map((run) => run.improvement));
    const finalStats = summarize(runs.map((run) => run.finalScore));
    const improvedCases = runs.filter((run) => run.priorityComparison < -1e-6).length;
    const regressions = runs.filter((run) => run.regressed).length;

    console.log(`\n--- ${label} aggregate ---`);
    console.log(`avg improvement: ${improveStats.mean.toFixed(2)}`);
    console.log(`improved cases: ${improvedCases}/${runs.length}`);
    console.log(
        `final score: mean=${finalStats.mean.toFixed(2)} p50=${finalStats.p50.toFixed(2)} p90=${finalStats.p90.toFixed(2)}`,
    );
    console.log(`no-regression violations: ${regressions}/${runs.length}`);
}

function printAdversarialTable(resultsByConfig: Map<string, ScenarioRun[]>): void {
    const adversarialNames = createAdversarialScenarios().map((scenario) => scenario.name);
    const headers = [
        'scenario'.padEnd(42),
        'start'.padStart(8),
        'baseline'.padStart(10),
        'archive'.padStart(10),
        'arch+adapt'.padStart(12),
        'combined'.padStart(10),
    ];
    console.log('\n=== Adversarial Before/After (final score) ===');
    console.log(headers.join(' | '));
    console.log('-'.repeat(headers.join(' | ').length));

    for (const name of adversarialNames) {
        const baseline = findScenario(resultsByConfig.get('baseline'), name);
        const archiveOnly = findScenario(resultsByConfig.get('archive_only'), name);
        const archiveAdaptive = findScenario(resultsByConfig.get('archive+adaptive_fixed'), name);
        const combined = findScenario(resultsByConfig.get('combined_fixed'), name);
        if (!baseline || !archiveOnly || !archiveAdaptive || !combined) continue;
        console.log(
            [
                name.slice(0, 42).padEnd(42),
                baseline.startScore.toFixed(1).padStart(8),
                baseline.finalScore.toFixed(1).padStart(10),
                archiveOnly.finalScore.toFixed(1).padStart(10),
                archiveAdaptive.finalScore.toFixed(1).padStart(12),
                combined.finalScore.toFixed(1).padStart(10),
            ].join(' | '),
        );
    }
}

function findScenario(runs: ScenarioRun[] | undefined, scenario: string): ScenarioRun | undefined {
    return runs?.find((run) => run.scenario === scenario);
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
    const resultsByConfig = new Map<string, ScenarioRun[]>();
    for (const namedConfig of CONFIGS) {
        const runs = await runConfig(namedConfig);
        resultsByConfig.set(namedConfig.name, runs);
        printAggregate(namedConfig.name, runs);
    }
    printAdversarialTable(resultsByConfig);
}

run().catch((error) => {
    console.error(error);
});
