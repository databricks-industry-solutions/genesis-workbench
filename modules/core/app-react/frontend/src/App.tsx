import { useEffect } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import { Layout } from '@/components/Layout'
import { HomePage } from '@/pages/Home'
import { ProfilePage } from '@/pages/Profile'
import { SettingsPage } from '@/pages/Settings'
import { MonitoringPage } from '@/pages/Monitoring'
import { SingleCellPage } from '@/pages/SingleCell'
import { LargeMoleculePage } from '@/pages/LargeMolecule'
import { SmallMoleculePage } from '@/pages/SmallMolecule'
import { GenomicsPage } from '@/pages/Genomics'
import { BootstrapGate } from '@/components/BootstrapGate'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { useThemeStore } from '@/stores/theme'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      refetchOnWindowFocus: false,
      retry: false,
    },
  },
})

export default function App() {
  // Sync the Tailwind `dark` class on <html> with the persisted theme store
  // so every component using `bg-background`/`text-foreground`/etc. picks
  // up the correct CSS variables on mount + on any toggle.
  const theme = useThemeStore((s) => s.theme)
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'dark') root.classList.add('dark')
    else root.classList.remove('dark')
  }, [theme])

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <BootstrapGate>
            <Routes>
              <Route element={<Layout />}>
                <Route index element={<HomePage />} />
                <Route
                  path="single-cell"
                  element={
                    <ErrorBoundary>
                      <SingleCellPage />
                    </ErrorBoundary>
                  }
                />
                <Route path="large-molecule" element={<LargeMoleculePage />} />
                <Route path="small-molecule" element={<SmallMoleculePage />} />
                <Route path="genomics" element={<GenomicsPage />} />
                <Route path="profile" element={<ProfilePage />} />
                <Route path="monitoring" element={<MonitoringPage />} />
                <Route path="settings" element={<SettingsPage />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </BootstrapGate>
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  )
}
