import {
    type GridState,
    type Port,
    type BeltSegment,
    type BeltPath,
    Direction,
    CellType,
    directionToDelta,
    oppositeDirection,
} from './types';
import { getPortExternalTile, isInBounds } from './grid';
import { MinHeap } from './minheap';

interface AStarNode {
    x: number;
    y: number;
    g: number;
    h: number;
    f: number;
    parent: AStarNode | null;
    direction: Direction | null;
}

function heuristic(ax: number, ay: number, bx: number, by: number): number {
    return Math.abs(ax - bx) + Math.abs(ay - by);
}

function nodeKey(x: number, y: number): string {
    return `${x},${y}`;
}

const DIRECTIONS: Direction[] = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT];

/**
 * Prevents two belts from running parallel on 2+ consecutive shared tiles.
 * Single-tile crossing is allowed.
 */
function wouldCauseParallelOverlap(
    grid: GridState,
    current: AStarNode,
    nx: number,
    ny: number,
    connectionId: string,
): boolean {
    const neighborCell = grid.cells[ny][nx];
    if (neighborCell.type !== CellType.BELT || neighborCell.beltConnectionIds.length === 0) return false;

    const currentCell = grid.cells[current.y][current.x];
    if (currentCell.type !== CellType.BELT || currentCell.beltConnectionIds.length === 0) return false;

    for (const existingConnId of neighborCell.beltConnectionIds) {
        if (existingConnId === connectionId) continue;
        if (currentCell.beltConnectionIds.includes(existingConnId)) return true;
    }
    return false;
}

/**
 * A* pathfinder using binary min-heap for performance.
 */
export function findBeltPath(
    grid: GridState,
    sourcePort: Port,
    targetPort: Port,
    connectionId: string,
): BeltPath | null {
    const start = getPortExternalTile(sourcePort);
    const end = getPortExternalTile(targetPort);

    if (!isInBounds(grid, start.x, start.y) || !isInBounds(grid, end.x, end.y)) return null;
    if (grid.cells[start.y][start.x].type === CellType.MACHINE) return null;
    if (grid.cells[end.y][end.x].type === CellType.MACHINE) return null;

    const startDir = sourcePort.approachDirection;

    const openSet = new MinHeap<AStarNode>((a, b) => a.f - b.f);
    const closedSet = new Set<string>();
    const gScores = new Map<string, number>();

    const startNode: AStarNode = {
        x: start.x,
        y: start.y,
        g: 0,
        h: heuristic(start.x, start.y, end.x, end.y),
        f: heuristic(start.x, start.y, end.x, end.y),
        parent: null,
        direction: startDir,
    };

    openSet.push(startNode);
    gScores.set(nodeKey(start.x, start.y), 0);

    while (openSet.size > 0) {
        const current = openSet.pop()!;
        const currentKey = nodeKey(current.x, current.y);

        if (current.x === end.x && current.y === end.y) {
            return reconstructPath(current, connectionId);
        }

        if (closedSet.has(currentKey)) continue;
        closedSet.add(currentKey);

        for (const dir of DIRECTIONS) {
            const delta = directionToDelta(dir);
            const nx = current.x + delta.dx;
            const ny = current.y + delta.dy;
            const neighborKey = nodeKey(nx, ny);

            if (closedSet.has(neighborKey)) continue;
            if (!isInBounds(grid, nx, ny)) continue;

            const cell = grid.cells[ny][nx];
            if (cell.type === CellType.MACHINE) continue;
            if (wouldCauseParallelOverlap(grid, current, nx, ny, connectionId)) continue;

            const isTurn = current.direction !== null && current.direction !== dir;
            const turnCost = isTurn ? 2 : 0;
            const crossingCost = cell.type === CellType.BELT ? 0.5 : 0;
            const tentativeG = current.g + 1 + turnCost + crossingCost;

            const existingG = gScores.get(neighborKey);
            if (existingG !== undefined && tentativeG >= existingG) continue;

            gScores.set(neighborKey, tentativeG);

            openSet.push({
                x: nx,
                y: ny,
                g: tentativeG,
                h: heuristic(nx, ny, end.x, end.y),
                f: tentativeG + heuristic(nx, ny, end.x, end.y),
                parent: current,
                direction: dir,
            });
        }
    }

    return null;
}

/**
 * Fast Manhattan distance estimate between two ports.
 * Used in Phase 1 of the optimizer instead of full A*.
 */
export function estimateBeltLength(sourcePort: Port, targetPort: Port): number {
    const s = getPortExternalTile(sourcePort);
    const t = getPortExternalTile(targetPort);
    return Math.abs(s.x - t.x) + Math.abs(s.y - t.y);
}

function reconstructPath(endNode: AStarNode, connectionId: string): BeltPath {
    const segments: BeltSegment[] = [];
    let current: AStarNode | null = endNode;

    while (current !== null) {
        segments.unshift({
            x: current.x,
            y: current.y,
            fromDirection: current.parent ? oppositeDirection(current.direction!) : null,
            toDirection: null,
        });
        current = current.parent;
    }

    for (let i = 0; i < segments.length - 1; i++) {
        const next = segments[i + 1];
        const dx = next.x - segments[i].x;
        const dy = next.y - segments[i].y;
        if (dx === 1) segments[i].toDirection = Direction.RIGHT;
        else if (dx === -1) segments[i].toDirection = Direction.LEFT;
        else if (dy === 1) segments[i].toDirection = Direction.DOWN;
        else if (dy === -1) segments[i].toDirection = Direction.UP;
    }

    return { connectionId, segments };
}

/** Apply a belt path to the grid cells */
export function applyBeltPath(grid: GridState, path: BeltPath): void {
    for (const seg of path.segments) {
        const cell = grid.cells[seg.y][seg.x];
        if (cell.type === CellType.EMPTY) cell.type = CellType.BELT;
        if (!cell.beltConnectionIds.includes(path.connectionId)) {
            cell.beltConnectionIds.push(path.connectionId);
        }
    }
    grid.beltPaths.set(path.connectionId, path);
}

/** Remove a belt path from the grid cells */
export function removeBeltPath(grid: GridState, connectionId: string): void {
    const path = grid.beltPaths.get(connectionId);
    if (!path) return;

    for (const seg of path.segments) {
        const cell = grid.cells[seg.y][seg.x];
        cell.beltConnectionIds = cell.beltConnectionIds.filter((id) => id !== connectionId);
        if (cell.beltConnectionIds.length === 0 && cell.type === CellType.BELT) {
            cell.type = CellType.EMPTY;
        }
    }
    grid.beltPaths.delete(connectionId);
}
