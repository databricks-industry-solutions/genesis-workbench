import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Drives a POST request whose response is a Server-Sent Events stream:
 *   event: progress    data: {"pct": int, "msg": string}
 *   event: result      data: <terminal payload>
 *   event: error       data: {"message": string}
 *
 * Designed to replace `useMutation` for the few workflows where we stream
 * real progress from the backend (e.g. SCimilarity + TEDDY cluster
 * annotation). Native `EventSource` is GET-only, so we use fetch + a
 * ReadableStream + manual SSE framing.
 */
export type SseProgress = { pct: number; msg: string }

export type UseSseMutationResult<TParams, TResult> = {
  start: (params: TParams) => void
  reset: () => void
  abort: () => void
  /** Imperatively seed result data (e.g. to restore a previously-persisted
   * value from MLflow on tab load). Pass null to clear. */
  setData: (value: TResult | null) => void
  isPending: boolean
  progress: SseProgress | null
  data: TResult | null
  error: Error | null
}

export function useSseMutation<TParams, TResult>(
  url: string,
): UseSseMutationResult<TParams, TResult> {
  const [isPending, setIsPending] = useState(false)
  const [progress, setProgress] = useState<SseProgress | null>(null)
  const [data, setData] = useState<TResult | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const ctrlRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    setIsPending(false)
    setProgress(null)
    setData(null)
    setError(null)
  }, [])

  const abort = useCallback(() => {
    ctrlRef.current?.abort()
    ctrlRef.current = null
    setIsPending(false)
  }, [])

  const start = useCallback(
    (params: TParams) => {
      ctrlRef.current?.abort()
      const ctrl = new AbortController()
      ctrlRef.current = ctrl
      setIsPending(true)
      setProgress(null)
      setData(null)
      setError(null)

      ;(async () => {
        try {
          const res = await fetch(url, {
            method: 'POST',
            credentials: 'include',
            headers: {
              'Content-Type': 'application/json',
              Accept: 'text/event-stream',
            },
            body: JSON.stringify(params),
            signal: ctrl.signal,
          })
          if (!res.ok || !res.body) {
            const text = await res.text().catch(() => '')
            throw new Error(`HTTP ${res.status} ${url}: ${text || res.statusText}`)
          }
          const reader = res.body.getReader()
          const decoder = new TextDecoder()
          let buf = ''
          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            buf += decoder.decode(value, { stream: true })
            let sep
            while ((sep = buf.indexOf('\n\n')) !== -1) {
              const frame = buf.slice(0, sep)
              buf = buf.slice(sep + 2)
              if (!frame || frame.startsWith(':')) continue // keepalive comment
              let event = 'message'
              let dataLine = ''
              for (const line of frame.split('\n')) {
                if (line.startsWith('event:')) event = line.slice(6).trim()
                else if (line.startsWith('data:')) dataLine += line.slice(5).trim()
              }
              if (!dataLine) continue
              let payload: unknown
              try {
                payload = JSON.parse(dataLine)
              } catch {
                continue
              }
              if (event === 'progress') {
                setProgress(payload as SseProgress)
              } else if (event === 'result') {
                setData(payload as TResult)
                setIsPending(false)
                ctrlRef.current = null
                return
              } else if (event === 'error') {
                const msg =
                  (payload as { message?: string })?.message ?? 'Unknown error'
                setError(new Error(msg))
                setIsPending(false)
                ctrlRef.current = null
                return
              }
            }
          }
          // Stream ended without a result/error event — unexpected.
          if (ctrlRef.current === ctrl) {
            setError(new Error('Stream closed before result arrived'))
            setIsPending(false)
            ctrlRef.current = null
          }
        } catch (e) {
          if ((e as Error).name === 'AbortError') return
          setError(e as Error)
          setIsPending(false)
          ctrlRef.current = null
        }
      })()
    },
    [url],
  )

  // Abort any in-flight stream on unmount so unmounted components don't
  // keep streaming bytes into orphan state.
  useEffect(() => () => ctrlRef.current?.abort(), [])

  return { start, reset, abort, setData, isPending, progress, data, error }
}
