import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { useUserStore } from '@/stores/user'

export function useBootstrap() {
  const setBootstrap = useUserStore((s) => s.setBootstrap)
  const query = useQuery({
    queryKey: ['bootstrap'],
    queryFn: api.bootstrap,
    staleTime: 5 * 60_000,
  })

  useEffect(() => {
    if (query.data) setBootstrap(query.data)
  }, [query.data, setBootstrap])

  return query
}
