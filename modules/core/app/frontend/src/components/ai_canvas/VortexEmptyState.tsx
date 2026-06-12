// Vortex — animated empty-canvas backdrop. An "accretion vortex": scattered pieces
// orbit the eye in its spin direction, spiral inward (each inner group scales toward
// the eye) and fade at the core, while staggered delays keep fresh pieces appearing
// at the rim. Pure CSS/SVG, brand-colored, reduced-motion-safe. Rendered only when
// the canvas has no nodes; pointer-events-none so it never blocks dropping a node.

const SPIN = 40 // seconds — slow inflow loop (orbit + draw-in share it)
const EYE_SPIN = 64 // seconds — the eye turns even slower

// keyframes scoped with a vtx- prefix so they can't clash with other animations.
// Peak opacity is intentionally low so this stays a faint, calm backdrop.
const KEYFRAMES = `
@keyframes vtxSpin { to { transform: rotate(360deg); } }
@keyframes vtxDrawIn {
  0%   { transform: scale(1);   opacity: 0; }
  12%  { opacity: .22; }
  78%  { opacity: .22; }
  100% { transform: scale(.04); opacity: 0; }
}
@media (prefers-reduced-motion: reduce) {
  .vtx-anim * { animation: none !important; }
}
`

type Kind = 'hex' | 'sq' | 'circ' | 'diam' | 'tri' | 'dot'

// shape · color · start position (SVG coords, eye = 260,160). The negative
// animation delay is derived from the index so the field is already full on first
// paint and stays evenly spaced no matter what SPIN is.
const PIECES: { s: Kind; c: string; x: number; y: number }[] = [
  { s: 'hex', c: '#14b8a6', x: 400, y: 150 },
  { s: 'sq', c: '#f59e0b', x: 388, y: 212 },
  { s: 'circ', c: '#ef4444', x: 350, y: 268 },
  { s: 'diam', c: '#2dd4bf', x: 286, y: 300 },
  { s: 'tri', c: '#64748b', x: 210, y: 292 },
  { s: 'hex', c: '#14b8a6', x: 146, y: 258 },
  { s: 'sq', c: '#f59e0b', x: 118, y: 206 },
  { s: 'circ', c: '#ef4444', x: 112, y: 150 },
  { s: 'diam', c: '#2dd4bf', x: 130, y: 96 },
  { s: 'dot', c: '#5eead4', x: 188, y: 52 },
  { s: 'sq', c: '#f59e0b', x: 262, y: 40 },
  { s: 'circ', c: '#ef4444', x: 336, y: 52 },
  { s: 'hex', c: '#14b8a6', x: 392, y: 96 },
  { s: 'diam', c: '#2dd4bf', x: 360, y: 235 },
]

function Shape({ kind, fill, x, y }: { kind: Kind; fill: string; x: number; y: number }) {
  const t = `translate(${x},${y})`
  switch (kind) {
    case 'hex':
      return <polygon points="0,-8 7,-4 7,4 0,8 -7,4 -7,-4" transform={t} fill={fill} />
    case 'sq':
      return <rect x={-7} y={-7} width={14} height={14} rx={3} transform={t} fill={fill} />
    case 'circ':
      return <circle r={7} transform={t} fill={fill} />
    case 'diam':
      return <polygon points="0,-8 8,0 0,8 -8,0" transform={t} fill={fill} />
    case 'tri':
      return <polygon points="0,-9 8,6 -8,6" transform={t} fill={fill} />
    case 'dot':
      return <circle r={5} transform={t} fill={fill} />
  }
}

const EYE = { transformOrigin: '260px 160px' } as const

export function VortexEmptyState() {
  return (
    <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2">
      <style>{KEYFRAMES}</style>
      <svg
        className="vtx-anim h-[62%] max-h-[340px] w-auto"
        viewBox="0 0 520 320"
        preserveAspectRatio="xMidYMid meet"
        aria-hidden="true"
      >
        <defs>
          <radialGradient id="vtxGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#14b8a6" stopOpacity={0.12} />
            <stop offset="100%" stopColor="#14b8a6" stopOpacity={0} />
          </radialGradient>
        </defs>

        {/* the eye: faint glow + two very slowly-rotating spiral arcs */}
        <g style={{ ...EYE, animation: `vtxSpin ${EYE_SPIN}s linear infinite` }}>
          <circle cx={260} cy={160} r={80} fill="url(#vtxGlow)" />
          <path d="M260,160 m0,-64 a64,64 0 1,1 -45,18" fill="none" stroke="#14b8a6" strokeOpacity={0.22} strokeWidth={1.5} />
          <path d="M260,160 m0,-40 a40,40 0 1,0 28,11" fill="none" stroke="#2dd4bf" strokeOpacity={0.18} strokeWidth={1.5} />
          {/* much smaller inner ring; dasharray gap rotated ~180° so it sits almost
              opposite the middle arc's (upper-right) gap → lower-left */}
          <circle
            cx={260}
            cy={160}
            r={18}
            fill="none"
            stroke="#5eead4"
            strokeOpacity={0.2}
            strokeWidth={1.5}
            strokeDasharray="84 29"
            transform="rotate(180 260 160)"
          />
        </g>

        {/* scattered pieces spiral inward and vanish; fresh ones appear at the rim.
            Delay derived from index → evenly staggered across the (slow) loop. */}
        {PIECES.map((p, i) => {
          const delay = `${(-i * SPIN) / PIECES.length}s`
          return (
            <g key={i} style={{ ...EYE, animation: `vtxSpin ${SPIN}s linear infinite`, animationDelay: delay }}>
              <g style={{ ...EYE, animation: `vtxDrawIn ${SPIN}s ease-in infinite`, animationDelay: delay }}>
                <Shape kind={p.s} fill={p.c} x={p.x} y={p.y} />
              </g>
            </g>
          )
        })}
      </svg>

      <div className="rounded-md border border-dashed border-border bg-card/70 px-4 py-2 text-center text-xs text-muted-foreground">
        Drag a node from the left palette, or describe a goal to generate a workflow.
      </div>
    </div>
  )
}
