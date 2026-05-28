import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import { Layout } from '@/components/Layout'
import { HomePage } from '@/pages/Home'
import { ProfilePage } from '@/pages/Profile'
import { SettingsPage } from '@/pages/Settings'
import { MonitoringPage } from '@/pages/Monitoring'
import { SingleCellPage } from '@/pages/SingleCell'
import { ProteinStudiesPage } from '@/pages/ProteinStudies'
import { SmallMoleculesPage } from '@/pages/SmallMolecules'
import { DiseaseBiologyPage } from '@/pages/DiseaseBiology'
import { BootstrapGate } from '@/components/BootstrapGate'
import { ErrorBoundary } from '@/components/ErrorBoundary'

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
                <Route path="protein-studies" element={<ProteinStudiesPage />} />
                <Route path="small-molecules" element={<SmallMoleculesPage />} />
                <Route path="disease-biology" element={<DiseaseBiologyPage />} />
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
