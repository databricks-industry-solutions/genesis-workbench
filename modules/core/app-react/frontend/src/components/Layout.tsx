import { NavLink, Outlet } from 'react-router-dom'

import { useUserStore, selectIsSetupDone, selectDisplayName } from '@/stores/user'
import { cn } from '@/lib/utils'

type NavItem = { to: string; label: string; module?: string }

const ALL_NAV: NavItem[] = [
  { to: '/', label: 'Home' },
  { to: '/single-cell', label: 'Single Cell', module: 'single_cell' },
  { to: '/protein-studies', label: 'Protein Studies', module: 'protein_studies' },
  { to: '/small-molecules', label: 'Small Molecules', module: 'small_molecule' },
  { to: '/disease-biology', label: 'Disease Biology', module: 'disease_biology' },
  { to: '/profile', label: 'Profile' },
  { to: '/monitoring', label: 'Monitoring' },
  { to: '/settings', label: 'Settings' },
]

export function Layout() {
  const bootstrap = useUserStore((s) => s.bootstrap)
  const setupDone = useUserStore(selectIsSetupDone)
  const displayName = useUserStore(selectDisplayName)

  const deployed = new Set(bootstrap?.deployed_modules ?? [])
  const visible = ALL_NAV.filter((n) => !n.module || deployed.has(n.module))

  return (
    <div className="flex h-full bg-background text-foreground">
      <aside className="flex w-60 flex-col border-r border-border bg-card">
        <div className="border-b border-border px-5 py-4">
          <img
            src="/gwb_logo.png"
            alt="Genesis Workbench"
            className="block h-auto w-full"
          />
        </div>
        <nav className="flex flex-1 flex-col gap-1 p-3">
          {visible.map((item) => {
            const showWarning = item.to === '/profile' && bootstrap && !setupDone
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  cn(
                    'flex items-center justify-between rounded-md px-3 py-2 text-sm transition-colors',
                    isActive
                      ? 'bg-accent text-foreground'
                      : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
                  )
                }
              >
                <span>{item.label}</span>
                {showWarning && (
                  <span
                    className="rounded-full bg-destructive/15 px-1.5 py-0.5 text-[10px] uppercase text-destructive"
                    title="Setup incomplete"
                  >
                    !
                  </span>
                )}
              </NavLink>
            )
          })}
        </nav>
        <div className="border-t border-border px-5 py-3 text-xs text-muted-foreground">
          {bootstrap ? (
            <>
              <div className="truncate text-foreground">{displayName}</div>
              <div className="truncate">{bootstrap.user.email ?? ''}</div>
            </>
          ) : (
            <div>Loading…</div>
          )}
        </div>
      </aside>

      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
