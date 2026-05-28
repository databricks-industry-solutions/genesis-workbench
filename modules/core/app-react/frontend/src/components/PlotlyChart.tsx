import { useEffect, useRef } from 'react'
import Plotly from 'plotly.js/dist/plotly'

type Data = Parameters<typeof Plotly.newPlot>[1]
type Layout = Parameters<typeof Plotly.newPlot>[2]
type Config = Parameters<typeof Plotly.newPlot>[3]

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

  // (Re)render whenever data / layout / config changes.
  useEffect(() => {
    const el = elRef.current
    if (!el) return
    const layoutObj = layout ?? {}
    const configObj = config ?? {}
    if (!initialisedRef.current) {
      Plotly.newPlot(el, data, layoutObj, configObj)
      initialisedRef.current = true
    } else {
      Plotly.react(el, data, layoutObj, configObj)
    }
  }, [data, layout, config])

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
