import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type Theme = 'light' | 'dark'

type ThemeState = {
  theme: Theme
  setTheme: (t: Theme) => void
  toggle: () => void
}

/**
 * Theme preference, persisted to localStorage under the `gwb-theme` key.
 * The provider in main.tsx subscribes and toggles the `dark` class on
 * `<html>` whenever the value changes, which flips the Tailwind dark-mode
 * tokens defined in index.css.
 */
export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      theme: 'dark',
      setTheme: (t) => set({ theme: t }),
      toggle: () => set({ theme: get().theme === 'dark' ? 'light' : 'dark' }),
    }),
    { name: 'gwb-theme' },
  ),
)
