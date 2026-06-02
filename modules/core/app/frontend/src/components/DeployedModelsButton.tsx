import { useState } from 'react'

import { DeployedModelsTab } from '@/components/DeployedModelsTab'
import { Drawer } from '@/components/Drawer'
import type { ModuleName } from '@/types/api'

export function DeployedModelsButton({ module }: { module: ModuleName }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary shadow-sm transition-colors hover:bg-primary/20"
        title="View endpoints, workflow models, and workflow packages for this module"
      >
        <span className="material-symbols-outlined text-[20px] leading-none">component_exchange</span>
        Models & Packages
      </button>
      <Drawer
        open={open}
        onClose={() => setOpen(false)}
        title="Models & Packages"
        width="max-w-5xl"
      >
        <DeployedModelsTab module={module} />
      </Drawer>
    </>
  )
}
