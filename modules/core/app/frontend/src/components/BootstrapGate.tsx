import type { ReactNode } from 'react'

import { useBootstrap } from '@/hooks/useBootstrap'

export function BootstrapGate({ children }: { children: ReactNode }) {
  const q = useBootstrap()

  if (q.isLoading) {
    return (
      <div className="flex h-full items-center justify-center bg-background text-sm text-muted-foreground">
        Loading Genesis Workbench…
      </div>
    )
  }

  if (q.error || !q.data) {
    return (
      <div className="flex h-full items-center justify-center bg-background p-8">
        <div className="max-w-xl rounded-md border border-destructive/40 bg-destructive/10 p-5 text-sm text-destructive">
          <div className="font-semibold">Failed to load workspace bootstrap.</div>
          <pre className="mt-2 overflow-auto whitespace-pre-wrap text-xs">
            {String(q.error ?? 'Unknown error')}
          </pre>
          <div className="mt-3 text-muted-foreground">
            Check that the catalog binding, schema, and SQL warehouse are accessible to the app
            service principal.
          </div>
        </div>
      </div>
    )
  }

  return <>{children}</>
}
