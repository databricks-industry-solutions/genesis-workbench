import { useEffect, useMemo, useRef } from 'react'
import Plotly from 'plotly.js/dist/plotly'

import { useThemeStore } from '@/stores/theme'

type Data = Parameters<typeof Plotly.newPlot>[1]
type Layout = Parameters<typeof Plotly.newPlot>[2]
type Config = Parameters<typeof Plotly.newPlot>[3]

// Theme-aware plotly defaults. Transparent backgrounds let the surrounding
// container (which uses the theme's --background / --card) show through, so
// the chart never paints a hardcoded dark slab on a light page.
function themedDefaults(isDark: boolean): Partial<Layout> {
  const fg = isDark ? '#fafafa' : '#0a0a0a'
  const grid = isDark ? '#525252' : '#a3a3a3'
  const zero = isDark ? '#737373' : '#737373'
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: fg },
    xaxis: { gridcolor: grid, zerolinecolor: zero, linecolor: grid, tickcolor: grid },
    yaxis: { gridcolor: grid, zerolinecolor: zero, linecolor: grid, tickcolor: grid },
    legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: fg } },
  }
}

type Props = {
  data: Data
  layout?: Partial<Layout>
  config?: Partial<Config>
  style?: React.CSSProperties
  className?: string
  /** When true, the chart resizes itself on window resize. */
  useResizeHandler?: boolean
}

/**
 * Minimal React wrapper around plotly.js. Replaces `react-plotly.js` (v2.6.0,
 * unmaintained, uses componentWillReceiveProps / componentWillUpdate which
 * React 19 removes — surfaced as React error #130 "Element type is invalid:
 * got object" inside the Plot subtree).
 *
 * API mirrors `react-plotly.js`: pass `data`, `layout`, `config` and the
 * component takes care of newPlot on mount, react on update, purge on unmount.
 */
export function PlotlyChart({
  data,
  layout,
  config,
  style,
  className,
  useResizeHandler,
}: Props) {
  const elRef = useRef<HTMLDivElement | null>(null)
  // Track whether we've initialized so we can choose between newPlot and react.
  const initialisedRef = useRef(false)
  const theme = useThemeStore((s) => s.theme)

  // Merge theme defaults under the caller's layout — caller wins for any
  // field they explicitly set, theme fills in everything else (bg, fonts,
  // gridlines). Sub-objects like xaxis/yaxis/font merge field-by-field so
  // an explicit `xaxis.title` from the caller doesn't drop the themed
  // gridcolor.
  const mergedLayout = useMemo<Partial<Layout>>(() => {
    const base = themedDefaults(theme === 'dark')
    const ov = (layout ?? {}) as Record<string, unknown>
    return {
      ...base,
      ...ov,
      xaxis: { ...base.xaxis, ...(ov.xaxis as object | undefined) },
      yaxis: { ...base.yaxis, ...(ov.yaxis as object | undefined) },
      font: { ...base.font, ...(ov.font as object | undefined) },
      legend: { ...base.legend, ...(ov.legend as object | undefined) },
    } as Partial<Layout>
  }, [layout, theme])

  // (Re)render whenever data / layout / config changes.
  useEffect(() => {
    const el = elRef.current
    if (!el) return
    const configObj = config ?? {}
    if (!initialisedRef.current) {
      Plotly.newPlot(el, data, mergedLayout, configObj)
      initialisedRef.current = true
    } else {
      Plotly.react(el, data, mergedLayout, configObj)
    }
  }, [data, mergedLayout, config])

  // Optional resize handler — match the react-plotly.js API.
  useEffect(() => {
    if (!useResizeHandler) return
    const el = elRef.current
    if (!el) return
    const onResize = () => Plotly.Plots.resize(el)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [useResizeHandler])

  // Purge plotly state on unmount so we don't leak DOM listeners / canvases.
  useEffect(() => {
    const el = elRef.current
    return () => {
      if (el) Plotly.purge(el)
      initialisedRef.current = false
    }
  }, [])

  return <div ref={elRef} className={className} style={style} />
}
