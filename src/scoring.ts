import { type GridState, CellType } from './types';

export interface ScoreBreakdown {
    totalBelts: number;
    boundingBoxArea: number;
    cornerCount: number;
    totalScore: number;
}

export const BELT_WEIGHT = 1.0;
export const AREA_WEIGHT = 0.5;
export const CORNER_WEIGHT = 0.3;
export const SCORE_EPSILON = 1e-6;

export function evaluateGrid(grid: GridState): ScoreBreakdown {
    let totalBelts = 0;
    let cornerCount = 0;
    let minX = grid.width;
    let minY = grid.height;
    let maxX = -1;
    let maxY = -1;

    // Count belt segments and find bounding box
    for (let y = 0; y < grid.height; y++) {
        for (let x = 0; x < grid.width; x++) {
            const cell = grid.cells[y][x];
            if (cell.type !== CellType.EMPTY) {
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }
    }

    // Count total belt segments across all paths
    for (const path of grid.beltPaths.values()) {
        totalBelts += path.segments.length;
        // Count corners (direction changes)
        for (let i = 1; i < path.segments.length; i++) {
            const prev = path.segments[i - 1];
            const curr = path.segments[i];
            if (prev.toDirection !== null && curr.toDirection !== null && prev.toDirection !== curr.toDirection) {
                cornerCount++;
            }
        }
    }

    const boundingBoxArea = maxX >= 0 ? (maxX - minX + 1) * (maxY - minY + 1) : 0;

    const totalScore =
        totalBelts * BELT_WEIGHT +
        boundingBoxArea * AREA_WEIGHT +
        cornerCount * CORNER_WEIGHT;

    return { totalBelts, boundingBoxArea, cornerCount, totalScore };
}

export function compareScoreBreakdownLexicographic(
    a: ScoreBreakdown,
    b: ScoreBreakdown,
    epsilon = SCORE_EPSILON,
): number {
    if (Math.abs(a.totalBelts - b.totalBelts) > epsilon) {
        return a.totalBelts - b.totalBelts;
    }
    if (Math.abs(a.boundingBoxArea - b.boundingBoxArea) > epsilon) {
        return a.boundingBoxArea - b.boundingBoxArea;
    }
    if (Math.abs(a.cornerCount - b.cornerCount) > epsilon) {
        return a.cornerCount - b.cornerCount;
    }
    return 0;
}

export function isLexicographicImprovement(
    candidate: ScoreBreakdown,
    baseline: ScoreBreakdown,
    epsilon = SCORE_EPSILON,
): boolean {
    return compareScoreBreakdownLexicographic(candidate, baseline, epsilon) < 0;
}
