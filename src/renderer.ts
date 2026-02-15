import {
    type GridState,
    type Machine,
    type BeltPath,
    type Connection,
    getOrientedDimensions,
} from './types';
import { getMachinePorts } from './grid';

const COLORS = {
    background: '#1a1a2e',
    gridLine: '#1f1f38',
    gridLineAccent: '#2a2a50',
    machineBody: '#2a2960',
    machineBorder: '#5b57d1',
    machineLabel: '#c0bfff',
    portInput: '#5bc0eb',
    portOutput: '#f4845f',
    selectedHighlight: '#ffd166',
    beltColors: ['#e8c547', '#f4845f', '#5bc0eb', '#7bc950', '#e056a0', '#56e0d0', '#c05beb', '#eb5b5b'],
};

export interface Camera {
    x: number;
    y: number;
    zoom: number;
}

export interface GhostState {
    gx: number;
    gy: number;
    type: import('./types').MachineType;
    orientation: import('./types').Orientation;
    valid: boolean;
}

export class Renderer {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private cellSize = 32;
    public camera: Camera = { x: 0, y: 0, zoom: 1 };
    private selectedMachineId: string | null = null;
    private ghost: GhostState | null = null;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d')!;
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    resizeCanvas(): void {
        this.canvas.width = this.canvas.parentElement?.clientWidth || window.innerWidth;
        this.canvas.height = this.canvas.parentElement?.clientHeight || window.innerHeight;
    }

    setSelectedMachine(id: string | null): void {
        this.selectedMachineId = id;
    }

    setGhost(ghost: GhostState | null): void {
        this.ghost = ghost;
    }

    screenToGrid(sx: number, sy: number): { gx: number; gy: number } {
        const s = this.cellSize * this.camera.zoom;
        return {
            gx: Math.floor((sx - this.camera.x) / s),
            gy: Math.floor((sy - this.camera.y) / s),
        };
    }

    gridToScreen(gx: number, gy: number): { sx: number; sy: number } {
        const s = this.cellSize * this.camera.zoom;
        return { sx: gx * s + this.camera.x, sy: gy * s + this.camera.y };
    }

    /** Center of a grid cell in screen coords */
    private cellCenter(gx: number, gy: number, size: number): { cx: number; cy: number } {
        return {
            cx: gx * size + this.camera.x + size / 2,
            cy: gy * size + this.camera.y + size / 2,
        };
    }

    render(grid: GridState): void {
        const { ctx, canvas } = this;
        const size = this.cellSize * this.camera.zoom;

        ctx.fillStyle = COLORS.background;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        this.drawGridLines(grid, size);

        // Draw belts UNDER machines
        for (const path of grid.beltPaths.values()) {
            this.drawBeltPath(path, size, grid);
        }

        // Draw machines ON TOP
        for (const machine of grid.machines.values()) {
            this.drawMachine(machine, size);
        }

        // Draw ghost preview ON TOP of everything
        if (this.ghost) {
            this.drawGhost(this.ghost, size);
        }
    }

    // ─── Grid Lines ────────────────────────────────────────

    private drawGridLines(grid: GridState, size: number): void {
        const { ctx, canvas, camera } = this;
        const startGx = Math.max(0, Math.floor(-camera.x / size));
        const startGy = Math.max(0, Math.floor(-camera.y / size));
        const endGx = Math.min(grid.width, Math.ceil((canvas.width - camera.x) / size));
        const endGy = Math.min(grid.height, Math.ceil((canvas.height - camera.y) / size));

        for (let gx = startGx; gx <= endGx; gx++) {
            const sx = gx * size + camera.x;
            ctx.strokeStyle = gx % 5 === 0 ? COLORS.gridLineAccent : COLORS.gridLine;
            ctx.lineWidth = gx % 5 === 0 ? 1 : 0.5;
            ctx.beginPath();
            ctx.moveTo(sx, startGy * size + camera.y);
            ctx.lineTo(sx, endGy * size + camera.y);
            ctx.stroke();
        }
        for (let gy = startGy; gy <= endGy; gy++) {
            const sy = gy * size + camera.y;
            ctx.strokeStyle = gy % 5 === 0 ? COLORS.gridLineAccent : COLORS.gridLine;
            ctx.lineWidth = gy % 5 === 0 ? 1 : 0.5;
            ctx.beginPath();
            ctx.moveTo(startGx * size + camera.x, sy);
            ctx.lineTo(endGx * size + camera.x, sy);
            ctx.stroke();
        }
    }

    // ─── Machine ───────────────────────────────────────────

    private drawMachine(machine: Machine, size: number): void {
        const { ctx, camera } = this;
        const { width, height } = getOrientedDimensions(machine);
        const isSelected = this.selectedMachineId === machine.id;

        const sx = machine.x * size + camera.x;
        const sy = machine.y * size + camera.y;
        const w = width * size;
        const h = height * size;
        const pad = 3;
        const cornerR = 6;

        // Rounded rect body (no glow — just border color change for selection)
        ctx.fillStyle = COLORS.machineBody;
        this.roundRect(sx + pad, sy + pad, w - pad * 2, h - pad * 2, cornerR);
        ctx.fill();

        // Border — thicker + golden when selected, no shadow/glow
        ctx.strokeStyle = isSelected ? COLORS.selectedHighlight : COLORS.machineBorder;
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        this.roundRect(sx + pad, sy + pad, w - pad * 2, h - pad * 2, cornerR);
        ctx.stroke();

        // Label
        const fontSize = Math.max(10, size * 0.38);
        ctx.fillStyle = COLORS.machineLabel;
        ctx.font = `bold ${fontSize}px 'JetBrains Mono', monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${machine.type} `, sx + w / 2, sy + h / 2 - size * 0.12);

        const subSize = Math.max(8, size * 0.22);
        ctx.font = `${subSize}px 'JetBrains Mono', monospace`;
        ctx.fillStyle = '#8080aa';
        ctx.fillText(machine.orientation, sx + w / 2, sy + h / 2 + size * 0.2);

        // Ports
        this.drawPorts(machine, size);
    }

    private drawPorts(machine: Machine, size: number): void {
        const { ctx } = this;
        const { inputs, outputs } = getMachinePorts(machine);
        const portRadius = Math.max(3, size * 0.2);

        for (const port of inputs) {
            const { cx, cy } = this.cellCenter(port.x, port.y, size);
            ctx.beginPath();
            ctx.arc(cx, cy, portRadius, 0, Math.PI * 2);
            ctx.fillStyle = COLORS.portInput;
            ctx.fill();
            // White inner dot
            ctx.beginPath();
            ctx.arc(cx, cy, portRadius * 0.4, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
        }

        for (const port of outputs) {
            const { cx, cy } = this.cellCenter(port.x, port.y, size);
            ctx.beginPath();
            ctx.arc(cx, cy, portRadius, 0, Math.PI * 2);
            ctx.fillStyle = COLORS.portOutput;
            ctx.fill();
            // Hollow inner
            ctx.beginPath();
            ctx.arc(cx, cy, portRadius * 0.45, 0, Math.PI * 2);
            ctx.fillStyle = COLORS.machineBody;
            ctx.fill();
        }
    }

    // ─── Belt Path ─────────────────────────────────────────

    private drawBeltPath(path: BeltPath, size: number, grid: GridState): void {
        const { ctx } = this;
        if (path.segments.length === 0) return;

        const colorIdx = Math.abs(hashCode(path.connectionId)) % COLORS.beltColors.length;
        const color = COLORS.beltColors[colorIdx];
        const beltWidth = Math.max(3, size * 0.22);

        // Build the full polyline including extensions into the ports
        const points = this.buildBeltPolyline(path, size, grid);
        if (points.length < 2) return;

        // Draw dark outline underneath for depth
        ctx.strokeStyle = 'rgba(0,0,0,0.4)';
        ctx.lineWidth = beltWidth + 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
        ctx.stroke();

        // Draw main belt line
        ctx.strokeStyle = color;
        ctx.lineWidth = beltWidth;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
        ctx.stroke();

        // Crossing indicators: draw small dots where this belt crosses other belts
        this.drawCrossingDots(path, size, grid, color, beltWidth);

        // Endpoint dots at source and target
        const dotR = Math.max(2.5, size * 0.12);

        // Source dot
        ctx.beginPath();
        ctx.arc(points[0].x, points[0].y, dotR, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Target arrow
        if (points.length >= 2) {
            const tip = points[points.length - 1];
            const prev = points[points.length - 2];
            const angle = Math.atan2(tip.y - prev.y, tip.x - prev.x);
            const arrowLen = size * 0.3;

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(tip.x + Math.cos(angle) * arrowLen, tip.y + Math.sin(angle) * arrowLen);
            ctx.lineTo(
                tip.x + Math.cos(angle + 2.4) * arrowLen * 0.6,
                tip.y + Math.sin(angle + 2.4) * arrowLen * 0.6
            );
            ctx.lineTo(
                tip.x + Math.cos(angle - 2.4) * arrowLen * 0.6,
                tip.y + Math.sin(angle - 2.4) * arrowLen * 0.6
            );
            ctx.closePath();
            ctx.fill();
        }
    }

    /**
     * Build polyline points for a belt, extending from the source port
     * through all belt segments and into the target port.
     * This makes belts visually connect directly to the port circles.
     */
    private buildBeltPolyline(
        path: BeltPath,
        size: number,
        grid: GridState,
    ): { x: number; y: number }[] {
        const points: { x: number; y: number }[] = [];

        // Find the connection to get port positions
        const conn: Connection | undefined = grid.connections.get(path.connectionId);

        // Extend backwards into SOURCE output port
        if (conn) {
            const srcMachine = grid.machines.get(conn.sourceMachineId);
            if (srcMachine) {
                const srcPorts = getMachinePorts(srcMachine);
                const srcPort = srcPorts.outputs[conn.sourcePortIndex];
                if (srcPort) {
                    const { cx, cy } = this.cellCenter(srcPort.x, srcPort.y, size);
                    points.push({ x: cx, y: cy });
                }
            }
        }

        // All belt segment centers
        for (const seg of path.segments) {
            const { cx, cy } = this.cellCenter(seg.x, seg.y, size);
            points.push({ x: cx, y: cy });
        }

        // Extend forwards into TARGET input port
        if (conn) {
            const tgtMachine = grid.machines.get(conn.targetMachineId);
            if (tgtMachine) {
                const tgtPorts = getMachinePorts(tgtMachine);
                const tgtPort = tgtPorts.inputs[conn.targetPortIndex];
                if (tgtPort) {
                    const { cx, cy } = this.cellCenter(tgtPort.x, tgtPort.y, size);
                    points.push({ x: cx, y: cy });
                }
            }
        }

        return points;
    }

    /**
     * At tiles where this belt shares space with other belts (crossings),
     * draw a small colored circle to make the crossing point visible.
     */
    private drawCrossingDots(
        path: BeltPath,
        size: number,
        grid: GridState,
        color: string,
        beltWidth: number,
    ): void {
        const { ctx } = this;

        for (const seg of path.segments) {
            const cell = grid.cells[seg.y]?.[seg.x];
            if (!cell || cell.beltConnectionIds.length < 2) continue;

            // This tile has multiple belts — it's a crossing point
            const { cx, cy } = this.cellCenter(seg.x, seg.y, size);
            const crossR = beltWidth * 0.9;

            // Draw a filled circle with the belt color + white ring
            ctx.beginPath();
            ctx.arc(cx, cy, crossR, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }

    // ─── Ghost Preview ───────────────────────────────────────

    private drawGhost(ghost: GhostState, size: number): void {
        const { ctx, camera } = this;

        // Build a temporary machine to get dimensions
        const tempMachine: Machine = {
            id: '__ghost__',
            type: ghost.type,
            x: ghost.gx,
            y: ghost.gy,
            orientation: ghost.orientation,
        };
        const { width, height } = getOrientedDimensions(tempMachine);

        const sx = ghost.gx * size + camera.x;
        const sy = ghost.gy * size + camera.y;
        const w = width * size;
        const h = height * size;
        const pad = 3;
        const cornerR = 6;

        ctx.globalAlpha = 0.45;

        // Fill — green if valid, red if invalid
        ctx.fillStyle = ghost.valid ? '#2a6040' : '#602a2a';
        this.roundRect(sx + pad, sy + pad, w - pad * 2, h - pad * 2, cornerR);
        ctx.fill();

        // Border
        ctx.strokeStyle = ghost.valid ? '#4ae080' : '#e04a4a';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        this.roundRect(sx + pad, sy + pad, w - pad * 2, h - pad * 2, cornerR);
        ctx.stroke();
        ctx.setLineDash([]);

        // Ghost ports
        const ports = getMachinePorts(tempMachine);
        const portR = Math.max(2, size * 0.14);

        for (const port of ports.inputs) {
            const { cx, cy } = this.cellCenter(port.x, port.y, size);
            ctx.beginPath();
            ctx.arc(cx, cy, portR, 0, Math.PI * 2);
            ctx.fillStyle = COLORS.portInput;
            ctx.fill();
        }
        for (const port of ports.outputs) {
            const { cx, cy } = this.cellCenter(port.x, port.y, size);
            ctx.beginPath();
            ctx.arc(cx, cy, portR, 0, Math.PI * 2);
            ctx.fillStyle = COLORS.portOutput;
            ctx.fill();
        }

        // Label
        ctx.fillStyle = '#fff';
        const fontSize = Math.max(9, size * 0.32);
        ctx.font = `bold ${fontSize}px 'JetBrains Mono', monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(ghost.type, sx + w / 2, sy + h / 2 - size * 0.08);

        const subSize = Math.max(7, size * 0.2);
        ctx.font = `${subSize}px 'JetBrains Mono', monospace`;
        ctx.fillStyle = '#aaa';
        ctx.fillText(ghost.orientation, sx + w / 2, sy + h / 2 + size * 0.18);

        ctx.globalAlpha = 1.0;
    }

    // ─── Utility ───────────────────────────────────────────

    private roundRect(x: number, y: number, w: number, h: number, r: number): void {
        const { ctx } = this;
        r = Math.min(r, w / 2, h / 2);
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath();
    }
}

function hashCode(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash |= 0;
    }
    return hash;
}
