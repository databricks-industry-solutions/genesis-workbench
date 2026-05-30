import { useState } from 'react'
import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

type Tab = { id: string; label: string; content: ReactNode }

export function Tabs({
  tabs,
  initial,
  rightAccessory,
}: {
  tabs: Tab[]
  initial?: string
  rightAccessory?: ReactNode
}) {
  const [active, setActive] = useState(initial ?? tabs[0]?.id)
  return (
    <div>
      <div className="mb-4 flex items-end justify-between gap-3 border-b border-border">
        <div className="flex gap-1">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setActive(t.id)}
              className={cn(
                'rounded-t-md px-4 py-2 text-sm transition-colors',
                active === t.id
                  ? 'border-b-2 border-red-600 font-bold text-red-600 dark:border-red-500 dark:text-red-500'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground',
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
        {rightAccessory && <div className="pb-1">{rightAccessory}</div>}
      </div>
      <div>{tabs.find((t) => t.id === active)?.content}</div>
    </div>
  )
}
