import {
  type GridState,
  type Machine,
  type Connection,
  type Port,
  MachineType,
  Orientation,
} from './types';
import { createGrid, placeMachine, removeMachine, getMachinePorts, generateId, canPlaceMachine } from './grid';
import { findBeltPath, applyBeltPath, removeBeltPath } from './pathfinder';
import { evaluateGrid } from './scoring';
import { Renderer } from './renderer';
import { runOptimizer } from './optimizer';
import './style.css';

// ─── State ───────────────────────────────────────────────

const GRID_SIZE = 50;
let grid: GridState = createGrid(GRID_SIZE, GRID_SIZE);
let selectedMachineId: string | null = null;
let selectedMachineType: MachineType = MachineType.M3x3;

type InteractionMode = 'place' | 'select' | 'connect' | 'delete';
let mode: InteractionMode = 'place';
type OptimizeMode = 'normal' | 'deep';
let isOptimizerRunning = false;
let deepSearchActive = false;
let stopDeepSearchRequested = false;

const DEEP_SEARCH_CHUNK_BUDGET_MS = 2200;
const DEEP_SEARCH_PLATEAU_STOP_MS = 120000;
const SCORE_EPSILON = 1e-6;
interface PersistedEliteArchiveEntry {
  machines: Machine[];
  connections: Connection[];
}
let deepSearchEliteArchive: PersistedEliteArchiveEntry[] = [];

// Connection-building state
let connectSourcePort: Port | null = null;

// Drag state
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let dragMachineOrigX = 0;
let dragMachineOrigY = 0;

// Pan state
let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let panCameraStartX = 0;
let panCameraStartY = 0;

// Ghost placement preview state
let placementOrientation: Orientation = Orientation.NORTH;
let ghostGridX = 0;
let ghostGridY = 0;

// ─── DOM Setup ───────────────────────────────────────────

const app = document.getElementById('app')!;
app.innerHTML = `
  <div id="toolbar">
    <div class="toolbar-group">
      <span class="toolbar-label">Mode</span>
      <button id="btn-place" class="tool-btn active" data-mode="place" title="Place Machine">
        <span class="icon">⊞</span> Place <kbd>P</kbd>
      </button>
      <button id="btn-select" class="tool-btn" data-mode="select" title="Select/Move">
        <span class="icon">⊡</span> Select <kbd>S</kbd>
      </button>
      <button id="btn-connect" class="tool-btn" data-mode="connect" title="Connect Ports">
        <span class="icon">⟷</span> Connect <kbd>C</kbd>
      </button>
      <button id="btn-delete" class="tool-btn" data-mode="delete" title="Delete">
        <span class="icon">✕</span> Delete <kbd>D</kbd>
      </button>
    </div>
    <div class="toolbar-divider"></div>
    <div class="toolbar-group">
      <span class="toolbar-label">Machine</span>
      <button id="btn-3x3" class="type-btn active" data-type="3x3">3×3 <kbd>1</kbd></button>
      <button id="btn-5x3" class="type-btn" data-type="5x3">5×3 <kbd>2</kbd></button>
      <button id="btn-5x5" class="type-btn" data-type="5x5">5×5 <kbd>3</kbd></button>
    </div>
    <div class="toolbar-divider"></div>
    <div class="toolbar-group">
      <button id="btn-rotate" class="tool-btn" title="Rotate">
        <span class="icon">↻</span> Rotate <kbd>R</kbd>
      </button>
    </div>
    <div class="toolbar-divider"></div>
    <div class="toolbar-group">
      <button id="btn-optimize" class="tool-btn optimize-btn" title="Auto-Arrange">
        <span class="icon">⚡</span> Optimize <kbd>O</kbd>
      </button>
      <button id="btn-search-deeper" class="tool-btn deep-search-btn" title="Run longer optimizer search">
        <span class="icon">🔍</span> Search Deeper
      </button>
      <button id="btn-stop-search" class="tool-btn stop-search-btn" title="Stop continuous deep search" disabled>
        <span class="icon">■</span> Stop
      </button>
    </div>
    <div class="toolbar-divider"></div>
    <div class="toolbar-group">
      <span class="toolbar-label">File</span>
      <button id="btn-export" class="tool-btn" title="Export Layout">
        <span class="icon">💾</span> Export
      </button>
      <button id="btn-import" class="tool-btn" title="Import Layout">
        <span class="icon">📂</span> Import
      </button>
      <input type="file" id="import-file" accept=".json" style="display:none" />
    </div>
    <div class="toolbar-divider"></div>
    <div id="score-panel">
      <span id="score-belts">Belts: 0</span>
      <span id="score-area">Area: 0</span>
      <span id="score-corners">Corners: 0</span>
      <span id="score-total">Score: 0</span>
    </div>
  </div>
  <div id="status-bar">
    <span id="status-text">Place mode — Click grid to place a machine</span>
    <span id="coord-text">0, 0</span>
  </div>
  <div id="canvas-container">
    <canvas id="grid-canvas"></canvas>
  </div>
`;

const canvas = document.getElementById('grid-canvas') as HTMLCanvasElement;
const statusText = document.getElementById('status-text')!;
const coordText = document.getElementById('coord-text')!;

const renderer = new Renderer(canvas);
// Center the grid a bit
renderer.camera.x = 40;
renderer.camera.y = 60;

// ─── Mode Buttons ────────────────────────────────────────

const modeButtons = document.querySelectorAll<HTMLButtonElement>('.tool-btn[data-mode]');
modeButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    setMode(btn.dataset.mode as InteractionMode);
  });
});

function setMode(newMode: InteractionMode): void {
  mode = newMode;
  connectSourcePort = null;
  renderer.setGhost(null); // Clear ghost on mode change
  modeButtons.forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  updateStatus();
  requestRender();
}

// ─── Machine Type Buttons ────────────────────────────────

const typeButtons = document.querySelectorAll<HTMLButtonElement>('.type-btn');
typeButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    selectMachineType(btn.dataset.type as MachineType);
  });
});

function selectMachineType(type: MachineType): void {
  selectedMachineType = type;
  typeButtons.forEach((b) => b.classList.toggle('active', b.dataset.type === type));
  // Auto-switch to place mode when picking a machine type
  setMode('place');
}

// ─── Rotate Button ───────────────────────────────────────

document.getElementById('btn-rotate')!.addEventListener('click', () => {
  rotateSelected();
});

// ─── Optimize Button ─────────────────────────────────────

document.getElementById('btn-optimize')!.addEventListener('click', () => {
  runAutoOptimize('normal');
});

document.getElementById('btn-search-deeper')!.addEventListener('click', () => {
  runAutoOptimize('deep');
});

document.getElementById('btn-stop-search')!.addEventListener('click', () => {
  requestDeepSearchStop();
});

// ─── Export / Import ─────────────────────────────────────

const importFileInput = document.getElementById('import-file') as HTMLInputElement;

document.getElementById('btn-export')!.addEventListener('click', () => {
  exportLayout();
});

document.getElementById('btn-import')!.addEventListener('click', () => {
  importFileInput.click();
});

importFileInput.addEventListener('change', () => {
  const file = importFileInput.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    importLayout(reader.result as string);
    importFileInput.value = ''; // Reset so same file can be re-imported
  };
  reader.readAsText(file);
});

interface LayoutData {
  version: 1;
  gridSize: number;
  machines: Machine[];
  connections: Connection[];
}

function exportLayout(): void {
  const data: LayoutData = {
    version: 1,
    gridSize: GRID_SIZE,
    machines: Array.from(grid.machines.values()),
    connections: Array.from(grid.connections.values()),
  };
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `factory-layout-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  statusText.textContent = `Exported ${grid.machines.size} machines, ${grid.connections.size} connections`;
}

function importLayout(json: string): void {
  try {
    const data = JSON.parse(json) as LayoutData;
    if (!data.machines || !data.connections) {
      statusText.textContent = 'Invalid layout file — missing machines or connections';
      return;
    }

    // Reset grid
    grid = createGrid(GRID_SIZE, GRID_SIZE);
    selectedMachineId = null;
    renderer.setSelectedMachine(null);

    // Place machines
    for (const m of data.machines) {
      placeMachine(grid, { ...m });
    }

    // Add connections and route belts
    for (const conn of data.connections) {
      grid.connections.set(conn.id, conn);
      const srcMachine = grid.machines.get(conn.sourceMachineId);
      const tgtMachine = grid.machines.get(conn.targetMachineId);
      if (!srcMachine || !tgtMachine) continue;

      const srcPorts = getMachinePorts(srcMachine);
      const tgtPorts = getMachinePorts(tgtMachine);
      const srcPort = srcPorts.outputs[conn.sourcePortIndex];
      const tgtPort = tgtPorts.inputs[conn.targetPortIndex];
      if (!srcPort || !tgtPort) continue;

      const path = findBeltPath(grid, srcPort, tgtPort, conn.id);
      if (path) applyBeltPath(grid, path);
    }

    updateScore();
    requestRender();
    statusText.textContent = `Imported ${data.machines.length} machines, ${data.connections.length} connections`;
  } catch {
    statusText.textContent = 'Failed to import — invalid JSON file';
  }
}



// ─── Keyboard Shortcuts ──────────────────────────────────

document.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return;
  switch (e.key.toLowerCase()) {
    case 'p': setMode('place'); break;
    case 's': setMode('select'); break;
    case 'c': setMode('connect'); break;
    case 'd': setMode('delete'); break;
    case 'r':
      if (mode === 'place') {
        // Rotate ghost preview
        rotatePlacementOrientation();
      } else {
        rotateSelected();
      }
      break;
    case 'o':
      runAutoOptimize(e.shiftKey ? 'deep' : 'normal');
      break;
    case '1': selectMachineType(MachineType.M3x3); break;
    case '2': selectMachineType(MachineType.M5x3); break;
    case '3': selectMachineType(MachineType.M5x5); break;
    case 'escape':
      if (deepSearchActive) {
        requestDeepSearchStop();
        break;
      }
      selectedMachineId = null;
      connectSourcePort = null;
      renderer.setSelectedMachine(null);
      renderer.setGhost(null);
      updateStatus();
      requestRender();
      break;
  }
});

// ─── Canvas Interactions ─────────────────────────────────

canvas.addEventListener('mousedown', (e) => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;

  // Middle click or right click = start pan
  if (e.button === 1 || e.button === 2) {
    isPanning = true;
    panStartX = e.clientX;
    panStartY = e.clientY;
    panCameraStartX = renderer.camera.x;
    panCameraStartY = renderer.camera.y;
    e.preventDefault();
    return;
  }

  const { gx, gy } = renderer.screenToGrid(sx, sy);

  switch (mode) {
    case 'place':
      handlePlace(gx, gy);
      break;
    case 'select':
      handleSelect(gx, gy, e);
      break;
    case 'connect':
      handleConnect(gx, gy);
      break;
    case 'delete':
      handleDelete(gx, gy);
      break;
  }
});

canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  const { gx, gy } = renderer.screenToGrid(sx, sy);
  coordText.textContent = `${gx}, ${gy}`;

  if (isPanning) {
    renderer.camera.x = panCameraStartX + (e.clientX - panStartX);
    renderer.camera.y = panCameraStartY + (e.clientY - panStartY);
    requestRender();
    return;
  }

  // Update ghost preview in place mode
  if (mode === 'place') {
    ghostGridX = gx;
    ghostGridY = gy;
    updateGhost();
    requestRender();
  }

  if (isDragging && selectedMachineId) {
    const machine = grid.machines.get(selectedMachineId);
    if (!machine) return;
    const effectiveSize = 32 * renderer.camera.zoom;
    const dx = Math.round((e.clientX - dragStartX) / effectiveSize);
    const dy = Math.round((e.clientY - dragStartY) / effectiveSize);
    const newX = dragMachineOrigX + dx;
    const newY = dragMachineOrigY + dy;

    if (newX !== machine.x || newY !== machine.y) {
      removeMachine(grid, machine.id);
      removeConnectionBelts(machine.id);
      machine.x = newX;
      machine.y = newY;
      if (!placeMachine(grid, machine)) {
        // Revert
        machine.x = dragMachineOrigX;
        machine.y = dragMachineOrigY;
        placeMachine(grid, machine);
      }
      rerouteConnectionBelts(machine.id);
      requestRender();
    }
  }
});

canvas.addEventListener('mouseup', () => {
  isDragging = false;
  isPanning = false;
});

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  const oldZoom = renderer.camera.zoom;
  const zoomDelta = e.deltaY > 0 ? 0.9 : 1.1;
  renderer.camera.zoom = Math.max(0.3, Math.min(3, renderer.camera.zoom * zoomDelta));

  // Zoom toward cursor
  renderer.camera.x = mx - (mx - renderer.camera.x) * (renderer.camera.zoom / oldZoom);
  renderer.camera.y = my - (my - renderer.camera.y) * (renderer.camera.zoom / oldZoom);

  requestRender();
}, { passive: false });

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

// ─── Interaction Handlers ────────────────────────────────

function handlePlace(gx: number, gy: number): void {
  const machine: Machine = {
    id: generateId(),
    type: selectedMachineType,
    x: gx,
    y: gy,
    orientation: placementOrientation,
  };

  if (placeMachine(grid, machine)) {
    selectedMachineId = machine.id;
    renderer.setSelectedMachine(machine.id);
    updateScore();
    requestRender();
  }
}

function handleSelect(gx: number, gy: number, e: MouseEvent): void {
  // Find machine at click position
  const cell = grid.cells[gy]?.[gx];
  if (cell?.machineId) {
    selectedMachineId = cell.machineId;
    renderer.setSelectedMachine(cell.machineId);

    // Start dragging
    isDragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    const machine = grid.machines.get(cell.machineId)!;
    dragMachineOrigX = machine.x;
    dragMachineOrigY = machine.y;
  } else {
    selectedMachineId = null;
    renderer.setSelectedMachine(null);
  }
  updateStatus();
  requestRender();
}

function handleConnect(gx: number, gy: number): void {
  // Find the port at this grid position
  const port = findPortAt(gx, gy);
  if (!port) {
    statusText.textContent = 'No port at this position';
    return;
  }

  if (!connectSourcePort) {
    // Must click an output port first
    if (port.portType !== 'OUTPUT') {
      statusText.textContent = 'Click an OUTPUT port (orange) first';
      return;
    }
    // Check if this output port is already used
    if (isPortInUse(port.machineId, port.index, 'output')) {
      statusText.textContent = 'This output port already has a connection';
      return;
    }
    connectSourcePort = port;
    statusText.textContent = `Selected output port ${port.index} — Now click an INPUT port`;
    return;
  }

  // Second click — must be an input port on a different machine
  if (port.portType !== 'INPUT') {
    statusText.textContent = 'Click an INPUT port (blue) to complete the connection';
    return;
  }
  if (port.machineId === connectSourcePort.machineId) {
    statusText.textContent = 'Cannot connect a machine to itself';
    return;
  }
  // Check if this input port is already used
  if (isPortInUse(port.machineId, port.index, 'input')) {
    statusText.textContent = 'This input port already has a connection';
    connectSourcePort = null;
    return;
  }

  const connection: Connection = {
    id: generateId(),
    sourceMachineId: connectSourcePort.machineId,
    sourcePortIndex: connectSourcePort.index,
    targetMachineId: port.machineId,
    targetPortIndex: port.index,
  };

  grid.connections.set(connection.id, connection);

  // Route belt
  const path = findBeltPath(grid, connectSourcePort, port, connection.id);
  if (path) {
    applyBeltPath(grid, path);
    statusText.textContent = `Connected! Belt length: ${path.segments.length}`;
  } else {
    statusText.textContent = 'No path found between these ports!';
    grid.connections.delete(connection.id);
  }

  connectSourcePort = null;
  updateScore();
  requestRender();
}

function handleDelete(gx: number, gy: number): void {
  const cell = grid.cells[gy]?.[gx];
  if (!cell) return;

  if (cell.machineId) {
    const machineId = cell.machineId;
    // Remove all connections involving this machine
    const toRemove: string[] = [];
    for (const [id, conn] of grid.connections) {
      if (conn.sourceMachineId === machineId || conn.targetMachineId === machineId) {
        toRemove.push(id);
      }
    }
    for (const id of toRemove) {
      removeBeltPath(grid, id);
      grid.connections.delete(id);
    }
    removeMachine(grid, machineId);
    if (selectedMachineId === machineId) {
      selectedMachineId = null;
      renderer.setSelectedMachine(null);
    }
    updateScore();
    requestRender();
  }
}

// ─── Ghost Preview ───────────────────────────────────────

function updateGhost(): void {
  if (mode !== 'place') {
    renderer.setGhost(null);
    return;
  }

  // Build temp machine to check validity
  const tempMachine: Machine = {
    id: '__ghost_check__',
    type: selectedMachineType,
    x: ghostGridX,
    y: ghostGridY,
    orientation: placementOrientation,
  };

  const valid = canPlaceMachine(grid, tempMachine);

  renderer.setGhost({
    gx: ghostGridX,
    gy: ghostGridY,
    type: selectedMachineType,
    orientation: placementOrientation,
    valid,
  });
}

function rotatePlacementOrientation(): void {
  const orientations = [Orientation.NORTH, Orientation.EAST, Orientation.SOUTH, Orientation.WEST];
  const idx = orientations.indexOf(placementOrientation);
  placementOrientation = orientations[(idx + 1) % 4];
  updateGhost();
  updateStatus();
  requestRender();
}

// ─── Helpers ─────────────────────────────────────────────

function findPortAt(gx: number, gy: number): Port | null {
  for (const machine of grid.machines.values()) {
    const { inputs, outputs } = getMachinePorts(machine);
    for (const port of [...inputs, ...outputs]) {
      if (port.x === gx && port.y === gy) return port;
    }
  }
  return null;
}

function isPortInUse(machineId: string, portIndex: number, side: 'input' | 'output'): boolean {
  for (const conn of grid.connections.values()) {
    if (side === 'input' && conn.targetMachineId === machineId && conn.targetPortIndex === portIndex) {
      return true;
    }
    if (side === 'output' && conn.sourceMachineId === machineId && conn.sourcePortIndex === portIndex) {
      return true;
    }
  }
  return false;
}

function rotateSelected(): void {
  if (!selectedMachineId) return;
  const machine = grid.machines.get(selectedMachineId);
  if (!machine) return;

  const orientations = [Orientation.NORTH, Orientation.EAST, Orientation.SOUTH, Orientation.WEST];
  const idx = orientations.indexOf(machine.orientation);
  const newOrientation = orientations[(idx + 1) % 4];

  // Remove, change orientation, re-place
  removeMachine(grid, machine.id);
  removeConnectionBelts(machine.id);
  machine.orientation = newOrientation;

  if (canPlaceMachine(grid, machine)) {
    placeMachine(grid, machine);
    rerouteConnectionBelts(machine.id);
  } else {
    // Revert
    machine.orientation = orientations[idx];
    placeMachine(grid, machine);
    rerouteConnectionBelts(machine.id);
  }
  updateScore();
  requestRender();
}

function removeConnectionBelts(machineId: string): void {
  for (const [id, conn] of grid.connections) {
    if (conn.sourceMachineId === machineId || conn.targetMachineId === machineId) {
      removeBeltPath(grid, id);
    }
  }
}

function rerouteConnectionBelts(machineId: string): void {
  for (const [id, conn] of grid.connections) {
    if (conn.sourceMachineId === machineId || conn.targetMachineId === machineId) {
      const srcMachine = grid.machines.get(conn.sourceMachineId);
      const tgtMachine = grid.machines.get(conn.targetMachineId);
      if (!srcMachine || !tgtMachine) continue;

      const srcPorts = getMachinePorts(srcMachine);
      const tgtPorts = getMachinePorts(tgtMachine);
      const srcPort = srcPorts.outputs[conn.sourcePortIndex];
      const tgtPort = tgtPorts.inputs[conn.targetPortIndex];
      if (!srcPort || !tgtPort) continue;

      const path = findBeltPath(grid, srcPort, tgtPort, id);
      if (path) {
        applyBeltPath(grid, path);
      }
    }
  }
}

async function runAutoOptimize(mode: OptimizeMode = 'normal'): Promise<void> {
  if (isOptimizerRunning) return;

  if (grid.machines.size === 0) {
    statusText.textContent = 'No machines to optimize!';
    return;
  }

  const optimizeBtn = document.getElementById('btn-optimize') as HTMLButtonElement;
  const searchDeeperBtn = document.getElementById('btn-search-deeper') as HTMLButtonElement;
  const stopSearchBtn = document.getElementById('btn-stop-search') as HTMLButtonElement;
  const startScore = evaluateGrid(grid).totalScore;
  const isDeep = mode === 'deep';
  if (isDeep) {
    deepSearchEliteArchive = [];
  }

  isOptimizerRunning = true;
  deepSearchActive = isDeep;
  stopDeepSearchRequested = false;
  optimizeBtn.disabled = true;
  searchDeeperBtn.disabled = true;
  stopSearchBtn.disabled = !isDeep;
  optimizeBtn.textContent = isDeep ? '⚡ Optimize' : '⚡ Optimizing...';
  searchDeeperBtn.textContent = isDeep ? '🔍 Searching...' : '🔍 Search Deeper';
  stopSearchBtn.textContent = '■ Stop';
  statusText.textContent = isDeep
    ? 'Search Deeper started — running continuously from current layout...'
    : 'Running optimizer...';

  try {
    if (!isDeep) {
      const result = await runOptimizer(
        grid,
        { mode: 'normal' },
        (best, iter, phase) => {
          statusText.textContent = `${phase} — iter ${iter} | belts: ${best.totalBelts} | area: ${best.boundingBoxArea} | score: ${best.totalScore.toFixed(1)}`;
        },
      );

      grid = result.grid;
      updateScore();
      statusText.textContent = `Optimization complete! ${result.iterations} iterations, score: ${result.score.totalScore.toFixed(1)}`;
      requestRender();
      return;
    }

    const startedAt = performance.now();
    let lastImprovementAt = startedAt;
    let bestScore = startScore;
    let totalIterations = 0;
    let cycles = 0;
    let autoStoppedOnPlateau = false;

    while (!stopDeepSearchRequested) {
      cycles++;
      const cycleNumber = cycles;
      const chunkConfig: Record<string, unknown> = {
        mode: 'deep',
        timeBudgetMs: DEEP_SEARCH_CHUNK_BUDGET_MS,
        useExplorationSeeds: true,
        phase1Restarts: 2,
        phase2Attempts: 2,
        localPolishPasses: 2,
        persistEliteArchive: true,
        incomingEliteArchive: deepSearchEliteArchive,
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

      const result = await runOptimizer(
        grid,
        chunkConfig,
        (best, iter, phase) => {
          const elapsedMs = performance.now() - startedAt;
          const idleMs = performance.now() - lastImprovementAt;
          const bestVisible = Math.min(bestScore, best.totalScore);
          statusText.textContent =
            `Search Deeper #${cycleNumber} — ${phase} iter ${iter} | best ${bestVisible.toFixed(1)} | elapsed ${formatDuration(elapsedMs)} | idle ${formatDuration(idleMs)}`;
        },
      );

      totalIterations += result.iterations;
      grid = result.grid;
      updateScore();
      requestRender();
      deepSearchEliteArchive = readOutgoingEliteArchive(chunkConfig.outgoingEliteArchive) ?? deepSearchEliteArchive;

      if (result.score.totalScore + SCORE_EPSILON < bestScore) {
        bestScore = result.score.totalScore;
        lastImprovementAt = performance.now();
      }

      const elapsedMs = performance.now() - startedAt;
      const idleMs = performance.now() - lastImprovementAt;

      if (idleMs >= DEEP_SEARCH_PLATEAU_STOP_MS) {
        autoStoppedOnPlateau = true;
        break;
      }

      statusText.textContent =
        `Search Deeper cycle ${cycleNumber} complete | best ${bestScore.toFixed(1)} | elapsed ${formatDuration(elapsedMs)} | idle ${formatDuration(idleMs)} | continuing...`;
    }

    const elapsedMs = performance.now() - startedAt;
    const improved = bestScore + SCORE_EPSILON < startScore;

    if (autoStoppedOnPlateau) {
      statusText.textContent = improved
        ? `Search Deeper auto-stopped (plateau) — improved ${startScore.toFixed(1)} → ${bestScore.toFixed(1)} in ${formatDuration(elapsedMs)}`
        : `Search Deeper auto-stopped (plateau) — no improvement after ${formatDuration(elapsedMs)} (best ${bestScore.toFixed(1)})`;
    } else if (stopDeepSearchRequested) {
      statusText.textContent = improved
        ? `Search Deeper stopped — improved ${startScore.toFixed(1)} → ${bestScore.toFixed(1)} in ${formatDuration(elapsedMs)}`
        : `Search Deeper stopped — no better layout found (best ${bestScore.toFixed(1)})`;
    } else {
      statusText.textContent = improved
        ? `Search Deeper complete — improved ${startScore.toFixed(1)} → ${bestScore.toFixed(1)} in ${totalIterations} iterations`
        : `Search Deeper complete — no better layout found (score ${bestScore.toFixed(1)})`;
    }
  } catch (err) {
    console.error(err);
    statusText.textContent = isDeep ? 'Search Deeper failed' : 'Optimization failed';
  } finally {
    isOptimizerRunning = false;
    deepSearchActive = false;
    stopDeepSearchRequested = false;
    deepSearchEliteArchive = [];
    optimizeBtn.disabled = false;
    searchDeeperBtn.disabled = false;
    stopSearchBtn.disabled = true;
    optimizeBtn.textContent = '⚡ Optimize';
    searchDeeperBtn.textContent = '🔍 Search Deeper';
    stopSearchBtn.textContent = '■ Stop';
  }
}

function requestDeepSearchStop(): void {
  if (!deepSearchActive || stopDeepSearchRequested) return;
  stopDeepSearchRequested = true;
  const stopSearchBtn = document.getElementById('btn-stop-search') as HTMLButtonElement;
  stopSearchBtn.disabled = true;
  stopSearchBtn.textContent = '■ Stopping...';
  statusText.textContent = 'Stopping Search Deeper after current chunk...';
}

function formatDuration(ms: number): string {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${sec.toString().padStart(2, '0')}`;
}

function readOutgoingEliteArchive(value: unknown): PersistedEliteArchiveEntry[] | null {
  if (!Array.isArray(value)) return null;
  const parsed: PersistedEliteArchiveEntry[] = [];
  for (const entry of value) {
    if (!entry || typeof entry !== 'object') continue;
    const maybeMachines = (entry as { machines?: unknown }).machines;
    const maybeConnections = (entry as { connections?: unknown }).connections;
    if (!Array.isArray(maybeMachines) || !Array.isArray(maybeConnections)) continue;
    parsed.push({
      machines: maybeMachines as Machine[],
      connections: maybeConnections as Connection[],
    });
  }
  return parsed;
}

function updateScore(): void {
  const score = evaluateGrid(grid);
  document.getElementById('score-belts')!.textContent = `Belts: ${score.totalBelts}`;
  document.getElementById('score-area')!.textContent = `Area: ${score.boundingBoxArea}`;
  document.getElementById('score-corners')!.textContent = `Corners: ${score.cornerCount}`;
  document.getElementById('score-total')!.textContent = `Score: ${score.totalScore.toFixed(1)}`;
}

function updateStatus(): void {
  switch (mode) {
    case 'place':
      statusText.textContent = `Place mode — ${selectedMachineType} [${placementOrientation}] — R to rotate, click to place`;
      break;
    case 'select':
      statusText.textContent = selectedMachineId
        ? `Selected machine — Drag to move, R to rotate`
        : `Select mode — Click a machine to select it`;
      break;
    case 'connect':
      statusText.textContent = connectSourcePort
        ? `Click an INPUT port (blue) to complete the connection`
        : `Connect mode — Click an OUTPUT port (orange) to start`;
      break;
    case 'delete':
      statusText.textContent = `Delete mode — Click a machine to delete it`;
      break;
  }
}

// ─── Render Loop ─────────────────────────────────────────

let renderRequested = false;

function requestRender(): void {
  if (renderRequested) return;
  renderRequested = true;
  requestAnimationFrame(() => {
    renderer.render(grid);
    renderRequested = false;
  });
}

// Initial render
requestRender();
