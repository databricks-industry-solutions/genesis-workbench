import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'

type Props = { children: ReactNode }
type State = { error: Error | null; info: ErrorInfo | null }

/**
 * Top-level boundary. Without this a crash inside a workflow tab makes the
 * whole app go white (React unmounts the tree, the prod build swallows the
 * console trace). This renders a copy-pasteable error card so we can see
 * what blew up.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, info: null }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    this.setState({ error, info })
    // eslint-disable-next-line no-console
    console.error('UI crash', error, info)
  }

  render(): ReactNode {
    if (!this.state.error) return this.props.children
    return (
      <div className="m-8 max-w-3xl rounded-md border border-destructive/40 bg-destructive/10 p-5 text-sm text-destructive">
        <div className="font-semibold">Something went wrong rendering this page.</div>
        <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap text-xs">
          {this.state.error.message}
          {'\n\n'}
          {this.state.error.stack ?? ''}
        </pre>
        {this.state.info?.componentStack && (
          <details className="mt-3 text-xs">
            <summary className="cursor-pointer">Component stack</summary>
            <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap">
              {this.state.info.componentStack}
            </pre>
          </details>
        )}
        <button
          onClick={() => this.setState({ error: null, info: null })}
          className="mt-3 rounded-md border border-destructive/40 px-3 py-1 text-xs hover:bg-destructive/20"
        >
          Dismiss and retry
        </button>
      </div>
    )
  }
}
