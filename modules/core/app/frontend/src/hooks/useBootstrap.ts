import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { useThemeStore } from '@/stores/theme'
import { useUserStore } from '@/stores/user'

export function useBootstrap() {
  const setBootstrap = useUserStore((s) => s.setBootstrap)
  const setTheme = useThemeStore((s) => s.setTheme)
  const query = useQuery({
    queryKey: ['bootstrap'],
    queryFn: api.bootstrap,
    staleTime: 5 * 60_000,
  })

  useEffect(() => {
    if (!query.data) return
    setBootstrap(query.data)
    // Hydrate the theme store from server-side user_settings — keeps the
    // preference consistent across devices/browsers. Falls back silently
    // to the localStorage-persisted value if the user hasn't set one.
    const serverTheme = query.data.user_settings?.theme
    if (serverTheme === 'dark' || serverTheme === 'light') {
      setTheme(serverTheme)
    }
  }, [query.data, setBootstrap, setTheme])

  return query
}
