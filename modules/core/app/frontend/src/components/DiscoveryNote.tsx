// A short, static "what this analysis lets you discover" callout shown at the
// top of each analysis tab — orients the scientist on the discovery value
// before they read the result (distinct from the per-result AI narrative).
import type { ReactNode } from 'react'

export function DiscoveryNote({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-md border border-sky-500/30 bg-sky-500/5 px-3 py-2 text-xs text-muted-foreground">
      <span className="font-medium text-sky-700 dark:text-sky-300">
        💡 What you can discover here:{' '}
      </span>
      {children}
    </div>
  )
}
