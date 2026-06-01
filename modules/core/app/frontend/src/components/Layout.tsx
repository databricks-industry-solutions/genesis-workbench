import { Fragment } from 'react'
import { NavLink, Outlet } from 'react-router-dom'

import { useUserStore, selectIsSetupDone, selectDisplayName } from '@/stores/user'
import { cn } from '@/lib/utils'

type NavItem = {
  to: string
  label: string
  icon: string
  module?: string
  /** Render a horizontal divider above this item — used to visually
   * separate workflow modules from admin/utility pages. */
  dividerAbove?: boolean
}

// Icon names map to Material Symbols (loaded via Google Fonts in index.html).
// Material Symbols ligature names — keep these stable; they map to the icons loaded in frontend/index.html.
const ALL_NAV: NavItem[] = [
  { to: '/', label: 'Home', icon: 'home' },
  { to: '/single-cell', label: 'Single Cell', icon: 'microbiology', module: 'single_cell' },
  { to: '/large-molecule', label: 'Large Molecule', icon: 'biotech', module: 'large_molecule' },
  { to: '/small-molecule', label: 'Small Molecule', icon: 'science', module: 'small_molecule' },
  { to: '/genomics', label: 'Genomics', icon: 'coronavirus', module: 'genomics' },
  { to: '/monitoring', label: 'Monitoring', icon: 'monitoring', dividerAbove: true },
  { to: '/settings', label: 'Settings', icon: 'settings' },
]

function MaterialIcon({ name, className }: { name: string; className?: string }) {
  return (
    <span
      aria-hidden
      className={cn('material-symbols-outlined text-[20px] leading-none', className)}
    >
      {name}
    </span>
  )
}

export function Layout() {
  const bootstrap = useUserStore((s) => s.bootstrap)
  const setupDone = useUserStore(selectIsSetupDone)
  const displayName = useUserStore(selectDisplayName)

  const deployed = new Set(bootstrap?.deployed_modules ?? [])
  const visible = ALL_NAV.filter((n) => !n.module || deployed.has(n.module))

  return (
    <div className="flex h-full bg-background text-foreground">
      <aside className="flex w-60 flex-col border-r border-border bg-card">
        <div className="border-b border-border bg-[hsl(0_0%_85%)] px-5 py-4 dark:bg-[hsl(0_0%_15%)]">
          <img
            src="/gwb_logo.png"
            alt="Genesis Workbench"
            className="block h-auto w-full"
          />
        </div>
        <nav className="flex flex-1 flex-col gap-1 p-3">
          {visible.map((item) => (
            <Fragment key={item.to}>
              {item.dividerAbove && (
                <hr className="my-2 border-t border-border" aria-hidden />
              )}
              <NavLink
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors',
                    isActive
                      ? 'bg-accent text-foreground'
                      : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
                  )
                }
              >
                <MaterialIcon name={item.icon} />
                <span>{item.label}</span>
              </NavLink>
            </Fragment>
          ))}
        </nav>
        <div className="space-y-2 border-t border-border px-5 py-3 text-xs text-muted-foreground">
          {bootstrap ? (
            <>
              <div>
                <div className="truncate text-foreground">{displayName}</div>
                <div className="truncate">{bootstrap.user.email ?? ''}</div>
              </div>
              <NavLink
                to="/profile"
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 rounded-md px-3 py-1.5 text-xs transition-colors',
                    isActive
                      ? 'bg-accent text-foreground'
                      : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
                  )
                }
              >
                <MaterialIcon name="account_circle" className="text-[18px]" />
                <span className="flex-1">Profile</span>
                {!setupDone && (
                  <span
                    className="rounded-full bg-destructive/15 px-1.5 py-0.5 text-[10px] uppercase text-destructive"
                    title="Setup incomplete"
                  >
                    !
                  </span>
                )}
              </NavLink>
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
