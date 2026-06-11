// Vortex — animated "AI thinking" sparkle for the generation dialog: three gold
// 4-point stars that scale big↔small on staggered timing (twinkle alternately).
// SVG + CSS (.gwb-twinkle); used only in the dialog, not the toolbar ✨ button.

// 4-point star path centered at (cx,cy): outer radius R, inner r.
function star(cx: number, cy: number, R: number, r: number): string {
  const pts: string[] = []
  for (let i = 0; i < 8; i++) {
    const a = (Math.PI / 4) * i
    const rad = i % 2 === 0 ? R : r
    pts.push(`${(cx + rad * Math.sin(a)).toFixed(1)},${(cy - rad * Math.cos(a)).toFixed(1)}`)
  }
  return `M${pts.join(' L')} Z`
}

const STARS = [
  { cx: 14, cy: 14, R: 9, r: 2.6, delay: '0s' },
  { cx: 23, cy: 7, R: 4.2, r: 1.3, delay: '0.43s' },
  { cx: 6, cy: 22, R: 3.6, r: 1.1, delay: '0.86s' },
]

export function GenerateSparkle({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" aria-hidden role="img">
      {STARS.map((s, i) => (
        <path
          key={i}
          className="gwb-twinkle"
          style={{ animationDelay: s.delay }}
          d={star(s.cx, s.cy, s.R, s.r)}
          fill="#facc15"
        />
      ))}
    </svg>
  )
}
