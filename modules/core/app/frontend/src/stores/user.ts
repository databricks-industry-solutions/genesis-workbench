import { create } from 'zustand'

import type { BootstrapResponse } from '@/types/api'

type UserState = {
  bootstrap: BootstrapResponse | null
  setBootstrap: (b: BootstrapResponse | null) => void
  setUserSettings: (s: Record<string, string>) => void
}

export const useUserStore = create<UserState>((set) => ({
  bootstrap: null,
  setBootstrap: (b) => set({ bootstrap: b }),
  setUserSettings: (s) =>
    set((state) =>
      state.bootstrap ? { bootstrap: { ...state.bootstrap, user_settings: s } } : state,
    ),
}))

export const selectIsSetupDone = (s: UserState): boolean =>
  s.bootstrap?.user_settings?.setup_done === 'Y'

export const selectDisplayName = (s: UserState): string =>
  s.bootstrap?.user_settings?.user_display_name ||
  s.bootstrap?.user?.display_name ||
  s.bootstrap?.user?.preferred_username ||
  s.bootstrap?.user?.email ||
  ''
