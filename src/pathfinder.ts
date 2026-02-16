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

interface OccupiedTileInfo {
    cornerCount: number;
    horizontalCount: number;
    verticalCount: number;
}

type Axis = 'H' | 'V';

function heuristic(ax: number, ay: number, bx: number, by: number): number {
    return Math.abs(ax - bx) + Math.abs(ay - by);
}

function stateKey(x: number, y: number, direction: Direction | null): string {
    return `${x},${y},${direction ?? 'NONE'}`;
}

function tileKey(x: number, y: number): string {
    return `${x},${y}`;
}

const DIRECTIONS: Direction[] = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT];

function directionToAxis(direction: Direction): Axis {
    return direction === Direction.LEFT || direction === Direction.RIGHT ? 'H' : 'V';
}

function isCornerSegment(segment: BeltSegment): boolean {
    if (segment.fromDirection === null || segment.toDirection === null) return false;
    return directionToAxis(segment.fromDirection) !== directionToAxis(segment.toDirection);
}

function getSegmentAxis(segment: BeltSegment): Axis | null {
    if (segment.fromDirection !== null && segment.toDirection !== null) {
        const fromAxis = directionToAxis(segment.fromDirection);
        const toAxis = directionToAxis(segment.toDirection);
        return fromAxis === toAxis ? fromAxis : null;
    }
    if (segment.fromDirection !== null) return directionToAxis(segment.fromDirection);
    if (segment.toDirection !== null) return directionToAxis(segment.toDirection);
    return null;
}

function createEmptyOccupiedTileInfo(): OccupiedTileInfo {
    return {
        cornerCount: 0,
        horizontalCount: 0,
        verticalCount: 0,
    };
}

function addSegmentToOccupiedTileInfo(info: OccupiedTileInfo, segment: BeltSegment): void {
    if (isCornerSegment(segment)) {
        info.cornerCount++;
        return;
    }
    const axis = getSegmentAxis(segment);
    if (axis === 'H') info.horizontalCount++;
    if (axis === 'V') info.verticalCount++;
}

function buildPathOccupiedInfo(path: BeltPath | undefined): Map<string, OccupiedTileInfo> {
    const info = new Map<string, OccupiedTileInfo>();
    if (!path) return info;
    for (const segment of path.segments) {
        const key = tileKey(segment.x, segment.y);
        let entry = info.get(key);
        if (!entry) {
            entry = createEmptyOccupiedTileInfo();
            info.set(key, entry);
        }
        addSegmentToOccupiedTileInfo(entry, segment);
    }
    return info;
}

function getEffectiveOccupiedTileInfo(
    grid: GridState,
    excludedByConnection: Map<string, OccupiedTileInfo>,
    x: number,
    y: number,
): OccupiedTileInfo | null {
    const key = tileKey(x, y);
    const base = grid.beltTileUsage.get(key);
    const excluded = excludedByConnection.get(key);
    const cornerCount = (base?.cornerCount ?? 0) - (excluded?.cornerCount ?? 0);
    const horizontalCount = (base?.horizontalCount ?? 0) - (excluded?.horizontalCount ?? 0);
    const verticalCount = (base?.verticalCount ?? 0) - (excluded?.verticalCount ?? 0);
    if (cornerCount <= 0 && horizontalCount <= 0 && verticalCount <= 0) return null;
    return { cornerCount, horizontalCount, verticalCount };
}

function hasAxisUsage(info: OccupiedTileInfo, axis: Axis): boolean {
    if (axis === 'H') return info.horizontalCount > 0;
    return info.verticalCount > 0;
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
    const excludedByConnection = buildPathOccupiedInfo(grid.beltPaths.get(connectionId));
    const startOccupied = getEffectiveOccupiedTileInfo(grid, excludedByConnection, start.x, start.y);
    const endOccupied = getEffectiveOccupiedTileInfo(grid, excludedByConnection, end.x, end.y);
    if ((startOccupied?.cornerCount ?? 0) > 0 || (endOccupied?.cornerCount ?? 0) > 0) return null;

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
    gScores.set(stateKey(start.x, start.y, startDir), 0);

    while (openSet.size > 0) {
        const current = openSet.pop()!;
        const currentStateKey = stateKey(current.x, current.y, current.direction);

        if (current.x === end.x && current.y === end.y) {
            return reconstructPath(current, connectionId);
        }

        if (closedSet.has(currentStateKey)) continue;
        closedSet.add(currentStateKey);

        for (const dir of DIRECTIONS) {
            const delta = directionToDelta(dir);
            const nx = current.x + delta.dx;
            const ny = current.y + delta.dy;
            const neighborStateKey = stateKey(nx, ny, dir);

            if (closedSet.has(neighborStateKey)) continue;
            if (!isInBounds(grid, nx, ny)) continue;

            const cell = grid.cells[ny][nx];
            if (cell.type === CellType.MACHINE) continue;

            const isTurn = current.direction !== null && current.direction !== dir;
            const currentOccupied = getEffectiveOccupiedTileInfo(grid, excludedByConnection, current.x, current.y);
            const neighborOccupied = getEffectiveOccupiedTileInfo(grid, excludedByConnection, nx, ny);
            const moveAxis = directionToAxis(dir);

            if (neighborOccupied) {
                if (neighborOccupied.cornerCount > 0) continue;
                if (hasAxisUsage(neighborOccupied, moveAxis)) continue;
            }

            if (currentOccupied) {
                if (isTurn) continue;
                if (hasAxisUsage(currentOccupied, moveAxis)) continue;
            }
            const turnCost = isTurn ? 2 : 0;
            const crossingCost = cell.type === CellType.BELT ? 0.5 : 0;
            const tentativeG = current.g + 1 + turnCost + crossingCost;

            const existingG = gScores.get(neighborStateKey);
            if (existingG !== undefined && tentativeG >= existingG) continue;

            gScores.set(neighborStateKey, tentativeG);

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

function updateTileUsageForSegment(
    grid: GridState,
    segment: BeltSegment,
    delta: 1 | -1,
): void {
    const key = tileKey(segment.x, segment.y);
    const current = grid.beltTileUsage.get(key) ?? createEmptyOccupiedTileInfo();
    if (isCornerSegment(segment)) {
        current.cornerCount += delta;
    } else {
        const axis = getSegmentAxis(segment);
        if (axis === 'H') current.horizontalCount += delta;
        if (axis === 'V') current.verticalCount += delta;
    }

    if (current.cornerCount <= 0 && current.horizontalCount <= 0 && current.verticalCount <= 0) {
        grid.beltTileUsage.delete(key);
        return;
    }

    current.cornerCount = Math.max(0, current.cornerCount);
    current.horizontalCount = Math.max(0, current.horizontalCount);
    current.verticalCount = Math.max(0, current.verticalCount);
    grid.beltTileUsage.set(key, current);
}

/** Apply a belt path to the grid cells */
export function applyBeltPath(grid: GridState, path: BeltPath): void {
    for (const seg of path.segments) {
        const cell = grid.cells[seg.y][seg.x];
        if (cell.type === CellType.EMPTY) cell.type = CellType.BELT;
        if (!cell.beltConnectionIds.includes(path.connectionId)) {
            cell.beltConnectionIds.push(path.connectionId);
        }
        updateTileUsageForSegment(grid, seg, 1);
    }
    grid.beltPaths.set(path.connectionId, path);
}

/** Remove a belt path from the grid cells */
export function removeBeltPath(grid: GridState, connectionId: string): void {
    const path = grid.beltPaths.get(connectionId);
    if (!path) return;

    for (const seg of path.segments) {
        updateTileUsageForSegment(grid, seg, -1);
        const cell = grid.cells[seg.y][seg.x];
        cell.beltConnectionIds = cell.beltConnectionIds.filter((id) => id !== connectionId);
        if (cell.beltConnectionIds.length === 0 && cell.type === CellType.BELT) {
            cell.type = CellType.EMPTY;
        }
    }
    grid.beltPaths.delete(connectionId);
}
