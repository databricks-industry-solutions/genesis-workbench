/**
 * Animated DNA double-helix loader for long-running AI calls.
 *
 * Visual: a row of horizontal rungs whose widths cascade through a sine-
 * like cycle (each rung shares the same animation but with a staggered
 * delay). The effect reads as a rotating double helix.
 */
export function DnaLoader({ label }: { label?: string }) {
  const rungs = 14
  const yStep = 6
  // Two colors alternated between rungs to suggest the two strands.
  const colors = ['hsl(var(--primary))', 'hsl(var(--success))']

  return (
    <div className="flex items-center gap-3">
      <svg
        viewBox={`0 0 40 ${rungs * yStep + 6}`}
        className="h-16 w-10 shrink-0"
        aria-hidden
      >
        {Array.from({ length: rungs }).map((_, i) => {
          const y = i * yStep + 6
          return (
            <line
              key={i}
              x1={2}
              y1={y}
              x2={38}
              y2={y}
              stroke={colors[i % 2]}
              strokeWidth={3}
              strokeLinecap="round"
              className="animate-dna-rung origin-center"
              // CSS-only stagger so rungs fire in sequence — gives the
              // rotation illusion. transform-box keeps the scale anchored
              // to each rung's own bounding box rather than the SVG root.
              style={{
                transformBox: 'fill-box',
                transformOrigin: 'center',
                animationDelay: `${i * 0.09}s`,
              }}
            />
          )
        })}
      </svg>
      {label && (
        <span className="text-sm text-muted-foreground">{label}</span>
      )}
    </div>
  )
}
