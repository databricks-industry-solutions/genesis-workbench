import { Fragment } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'

import { useUserStore, selectIsSetupDone, selectDisplayName } from '@/stores/user'
import { useThemeStore } from '@/stores/theme'
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

// Material Symbols ligature names — keep these stable; they map to the icons loaded in frontend/index.html.
const ALL_NAV: NavItem[] = [
  { to: '/', label: 'Home', icon: 'home' },
  { to: '/single-cell', label: 'Single Cell', icon: 'microbiology', module: 'single_cell' },
  { to: '/large-molecule', label: 'Large Molecule', icon: 'biotech', module: 'large_molecule' },
  { to: '/small-molecule', label: 'Small Molecule', icon: 'science', module: 'small_molecule' },
  { to: '/genomics', label: 'Genomics', icon: 'coronavirus', module: 'genomics' },
  { to: '/bionemo', label: 'BioNeMo', icon: 'genetics', module: 'bionemo' },
  { to: '/monitoring', label: 'Monitoring', icon: 'monitoring', dividerAbove: true },
  { to: '/settings', label: 'Settings', icon: 'settings' },
]

// Title shown on the right side of the slim top strip per route.
const ROUTE_TITLES: Record<string, string> = {
  '/': 'Home',
  '/single-cell': 'Single Cell Studies',
  '/large-molecule': 'Large Molecule',
  '/small-molecule': 'Small Molecule',
  '/genomics': 'Genomics',
  '/bionemo': 'NVIDIA BioNeMo',
  '/monitoring': 'Monitoring',
  '/settings': 'Settings',
  '/profile': 'Profile',
}

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
  const theme = useThemeStore((s) => s.theme)
  const location = useLocation()

  const deployed = new Set(bootstrap?.deployed_modules ?? [])
  const visible = ALL_NAV.filter((n) => !n.module || deployed.has(n.module))

  const sidebarBg = theme === 'dark' ? '/gwb_sidebar_dark.svg' : '/gwb_sidebar_light.svg'
  const pageTitle = ROUTE_TITLES[location.pathname] ?? ''

  return (
    <div className="flex h-full bg-background text-foreground">
      {/* Sidebar: animated SVG runs vertically behind logo + nav. No internal separators.
          Light theme gets a solid teal base (so the SVG's translucent gradient and pastel
          icons read against a real teal surface, not over the white page bg). Dark theme
          stays transparent — its SVG renders directly over the app's dark slate. */}
      <aside className="relative flex w-72 shrink-0 flex-col overflow-hidden border-r border-border bg-[#008080] dark:bg-transparent">
        <img
          src={sidebarBg}
          alt=""
          aria-hidden
          className="absolute inset-0 h-full w-full object-cover"
        />

        {/* Logo (no separator below — flows directly into nav over the SVG) */}
        <div className="relative z-10 px-5 pt-5 pb-3">
          <img
            src="/gwb_logo.png"
            alt="Genesis Workbench"
            className="block h-auto w-full"
          />
        </div>

        <nav className="relative z-10 flex flex-1 flex-col gap-1 px-3 pb-3">
          {visible.map((item) => (
            <Fragment key={item.to}>
              {item.dividerAbove && (
                <hr className="my-2 border-t border-border/40" aria-hidden />
              )}
              <NavLink
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 rounded-md px-3 py-2 text-sm text-white transition-colors',
                    isActive ? 'bg-white/20 font-bold' : 'font-medium hover:bg-white/10',
                  )
                }
              >
                <MaterialIcon name={item.icon} />
                <span>{item.label}</span>
              </NavLink>
            </Fragment>
          ))}
        </nav>

        <div className="relative z-10 space-y-2 border-t border-white/15 px-5 py-3 text-xs text-white/80">
          {bootstrap ? (
            <>
              <div>
                <div className="truncate font-medium text-white">{displayName}</div>
                <div className="truncate text-white/70">{bootstrap.user.email ?? ''}</div>
              </div>
              <NavLink
                to="/profile"
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 rounded-md px-3 py-1.5 text-xs text-white transition-colors',
                    isActive ? 'bg-white/20 font-bold' : 'hover:bg-white/10',
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

      {/* Main column: slim title strip on top (no border), page content below. */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <div className="flex shrink-0 items-center justify-start px-7 pt-5 pb-0">
          <h1 className="text-xl font-bold text-black dark:text-white">{pageTitle}</h1>
        </div>
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
