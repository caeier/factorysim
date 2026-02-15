import {
    type GridState,
    type GridCell,
    type Machine,
    type Port,
    CellType,
    PortType,
    Orientation,
    getOrientedDimensions,
    getPortCount,
    orientationToInputDirection,
    orientationToOutputDirection,
    directionToDelta,
} from './types';

let nextId = 1;
export function generateId(): string {
    return `id_${nextId++}`;
}

export function createGrid(width: number, height: number): GridState {
    const cells: GridCell[][] = [];
    for (let y = 0; y < height; y++) {
        cells[y] = [];
        for (let x = 0; x < width; x++) {
            cells[y][x] = { type: CellType.EMPTY, beltConnectionIds: [] };
        }
    }
    return {
        width,
        height,
        cells,
        machines: new Map(),
        connections: new Map(),
        beltPaths: new Map(),
    };
}

export function isInBounds(grid: GridState, x: number, y: number): boolean {
    return x >= 0 && y >= 0 && x < grid.width && y < grid.height;
}

export function canPlaceMachine(grid: GridState, machine: Machine, ignoreId?: string): boolean {
    const { width, height } = getOrientedDimensions(machine);
    for (let dy = 0; dy < height; dy++) {
        for (let dx = 0; dx < width; dx++) {
            const gx = machine.x + dx;
            const gy = machine.y + dy;
            if (!isInBounds(grid, gx, gy)) return false;
            const cell = grid.cells[gy][gx];
            if (cell.type === CellType.MACHINE && cell.machineId !== ignoreId) return false;
        }
    }
    return true;
}

export function placeMachine(grid: GridState, machine: Machine): boolean {
    if (!canPlaceMachine(grid, machine)) return false;
    const { width, height } = getOrientedDimensions(machine);
    for (let dy = 0; dy < height; dy++) {
        for (let dx = 0; dx < width; dx++) {
            const gx = machine.x + dx;
            const gy = machine.y + dy;
            grid.cells[gy][gx] = { type: CellType.MACHINE, machineId: machine.id, beltConnectionIds: [] };
        }
    }
    grid.machines.set(machine.id, machine);
    return true;
}

export function removeMachine(grid: GridState, machineId: string): void {
    const machine = grid.machines.get(machineId);
    if (!machine) return;
    const { width, height } = getOrientedDimensions(machine);
    for (let dy = 0; dy < height; dy++) {
        for (let dx = 0; dx < width; dx++) {
            const gx = machine.x + dx;
            const gy = machine.y + dy;
            grid.cells[gy][gx] = { type: CellType.EMPTY, beltConnectionIds: [] };
        }
    }
    grid.machines.delete(machineId);
}

export function getMachinePorts(machine: Machine): { inputs: Port[]; outputs: Port[] } {
    const { width, height } = getOrientedDimensions(machine);
    const portCount = getPortCount(machine);
    const inputDir = orientationToInputDirection(machine.orientation);
    const outputDir = orientationToOutputDirection(machine.orientation);
    const inputApproach = inputDir; // belt must come FROM this direction
    const outputApproach = outputDir; // belt must leave in this direction

    const inputs: Port[] = [];
    const outputs: Port[] = [];

    // Compute port positions based on orientation
    for (let i = 0; i < portCount; i++) {
        let ix: number, iy: number, ox: number, oy: number;

        switch (machine.orientation) {
            case Orientation.NORTH:
                // Input on north face (top row), output on south face (bottom row)
                ix = machine.x + i;
                iy = machine.y;
                ox = machine.x + i;
                oy = machine.y + height - 1;
                break;
            case Orientation.SOUTH:
                // Input on south face (bottom row), output on north face (top row)
                ix = machine.x + i;
                iy = machine.y + height - 1;
                ox = machine.x + i;
                oy = machine.y;
                break;
            case Orientation.EAST:
                // Input on east face (right col), output on west face (left col)
                ix = machine.x + width - 1;
                iy = machine.y + i;
                ox = machine.x;
                oy = machine.y + i;
                break;
            case Orientation.WEST:
                // Input on west face (left col), output on east face (right col)
                ix = machine.x;
                iy = machine.y + i;
                ox = machine.x + width - 1;
                oy = machine.y + i;
                break;
        }

        inputs.push({
            machineId: machine.id,
            portType: PortType.INPUT,
            index: i,
            x: ix,
            y: iy,
            approachDirection: inputApproach,
        });

        outputs.push({
            machineId: machine.id,
            portType: PortType.OUTPUT,
            index: i,
            x: ox,
            y: oy,
            approachDirection: outputApproach,
        });
    }

    return { inputs, outputs };
}

/** Get the tile just outside a port (where a belt would start/end) */
export function getPortExternalTile(port: Port): { x: number; y: number } {
    const delta = directionToDelta(port.approachDirection);
    return { x: port.x + delta.dx, y: port.y + delta.dy };
}

export function isMachineTile(grid: GridState, x: number, y: number): boolean {
    if (!isInBounds(grid, x, y)) return false;
    return grid.cells[y][x].type === CellType.MACHINE;
}

export function cloneGridState(grid: GridState): GridState {
    const cells: GridCell[][] = [];
    for (let y = 0; y < grid.height; y++) {
        cells[y] = [];
        for (let x = 0; x < grid.width; x++) {
            const c = grid.cells[y][x];
            cells[y][x] = {
                type: c.type,
                machineId: c.machineId,
                beltConnectionIds: [...c.beltConnectionIds],
            };
        }
    }
    return {
        width: grid.width,
        height: grid.height,
        cells,
        machines: new Map(
            Array.from(grid.machines.entries()).map(([k, v]) => [k, { ...v }])
        ),
        connections: new Map(
            Array.from(grid.connections.entries()).map(([k, v]) => [k, { ...v }])
        ),
        beltPaths: new Map(
            Array.from(grid.beltPaths.entries()).map(([k, v]) => [
                k,
                { connectionId: v.connectionId, segments: v.segments.map((s) => ({ ...s })) },
            ])
        ),
    };
}
