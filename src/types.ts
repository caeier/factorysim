// ─── Enums ───────────────────────────────────────────────

export enum MachineType {
  M3x3 = '3x3',
  M5x3 = '5x3',
  M5x5 = '5x5',
}

/** Orientation = which direction the INPUT ports face */
export enum Orientation {
  NORTH = 'NORTH',
  EAST = 'EAST',
  SOUTH = 'SOUTH',
  WEST = 'WEST',
}

export enum PortType {
  INPUT = 'INPUT',
  OUTPUT = 'OUTPUT',
}

export enum CellType {
  EMPTY = 'EMPTY',
  MACHINE = 'MACHINE',
  BELT = 'BELT',
}

export enum Direction {
  UP = 'UP',
  DOWN = 'DOWN',
  LEFT = 'LEFT',
  RIGHT = 'RIGHT',
}

// ─── Machine dimensions lookup ───────────────────────────

export interface MachineDimensions {
  baseWidth: number;
  baseHeight: number;
}

export const MACHINE_DIMS: Record<MachineType, MachineDimensions> = {
  [MachineType.M3x3]: { baseWidth: 3, baseHeight: 3 },
  [MachineType.M5x3]: { baseWidth: 5, baseHeight: 3 },
  [MachineType.M5x5]: { baseWidth: 5, baseHeight: 5 },
};

// ─── Core types ──────────────────────────────────────────

export interface Machine {
  id: string;
  type: MachineType;
  x: number; // top-left grid x
  y: number; // top-left grid y
  orientation: Orientation;
}

export interface Port {
  machineId: string;
  portType: PortType;
  index: number; // 0-based index along the port face
  x: number; // absolute grid x of this port
  y: number; // absolute grid y of this port
  /** Direction a belt must approach FROM to connect to this port */
  approachDirection: Direction;
}

export interface Connection {
  id: string;
  sourceMachineId: string;
  sourcePortIndex: number;
  targetMachineId: string;
  targetPortIndex: number;
}

export interface BeltSegment {
  x: number;
  y: number;
  fromDirection: Direction | null; // null for first segment
  toDirection: Direction | null; // null for last segment
}

export interface BeltPath {
  connectionId: string;
  segments: BeltSegment[];
}

export interface GridCell {
  type: CellType;
  machineId?: string;
  beltConnectionIds: string[]; // can have multiple belts crossing
}

export interface GridState {
  width: number;
  height: number;
  cells: GridCell[][];
  machines: Map<string, Machine>;
  connections: Map<string, Connection>;
  beltPaths: Map<string, BeltPath>;
}

// ─── Helper ──────────────────────────────────────────────

export function getOrientedDimensions(machine: Machine): { width: number; height: number } {
  const dims = MACHINE_DIMS[machine.type];
  if (machine.orientation === Orientation.EAST || machine.orientation === Orientation.WEST) {
    return { width: dims.baseHeight, height: dims.baseWidth };
  }
  return { width: dims.baseWidth, height: dims.baseHeight };
}

export function getPortCount(machine: Machine): number {
  const dims = MACHINE_DIMS[machine.type];
  // Ports are on the width face (the wider dimension)
  return dims.baseWidth;
}

export function oppositeDirection(dir: Direction): Direction {
  switch (dir) {
    case Direction.UP: return Direction.DOWN;
    case Direction.DOWN: return Direction.UP;
    case Direction.LEFT: return Direction.RIGHT;
    case Direction.RIGHT: return Direction.LEFT;
  }
}

export function directionToDelta(dir: Direction): { dx: number; dy: number } {
  switch (dir) {
    case Direction.UP: return { dx: 0, dy: -1 };
    case Direction.DOWN: return { dx: 0, dy: 1 };
    case Direction.LEFT: return { dx: -1, dy: 0 };
    case Direction.RIGHT: return { dx: 1, dy: 0 };
  }
}

export function orientationToInputDirection(orientation: Orientation): Direction {
  switch (orientation) {
    case Orientation.NORTH: return Direction.UP;
    case Orientation.EAST: return Direction.RIGHT;
    case Orientation.SOUTH: return Direction.DOWN;
    case Orientation.WEST: return Direction.LEFT;
  }
}

export function orientationToOutputDirection(orientation: Orientation): Direction {
  return oppositeDirection(orientationToInputDirection(orientation));
}
