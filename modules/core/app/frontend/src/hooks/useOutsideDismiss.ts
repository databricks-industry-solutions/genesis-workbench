import { useEffect, useRef, type RefObject } from 'react'

/**
 * Dismiss a popover/dropdown on an outside click or the Escape key.
 *
 * Single source of truth for the "click-away to close" behavior — use this for
 * every lightweight popover instead of hand-rolling a document listener per
 * component (modals use Dialog/Drawer instead).
 *
 * Implementation notes that matter:
 *  - CAPTURE-phase `pointerdown`: fires before any ancestor that calls
 *    `stopPropagation()` on a bubble-phase mousedown could swallow it. A
 *    bubble-phase listener can be silently blocked by such an ancestor, which
 *    leaves the popover stuck open on outside clicks (only Esc closing it) —
 *    the exact bug this hook exists to prevent.
 *  - Clicks INSIDE `ref` (including the trigger button, if it's within the
 *    container) don't close — the trigger's own handler toggles.
 *  - Listeners attach only while `enabled` (the popover is open).
 *  - `onClose` is read through a ref, so passing an inline arrow doesn't
 *    re-subscribe the listeners on every render.
 *
 * @param ref      the popover container; clicks within it are ignored
 * @param onClose  invoked on an outside click or Escape
 * @param enabled  whether the popover is currently open
 */
export function useOutsideDismiss(
  ref: RefObject<HTMLElement | null>,
  onClose: () => void,
  enabled: boolean,
): void {
  const cb = useRef(onClose)
  cb.current = onClose

  useEffect(() => {
    if (!enabled) return
    const onDown = (e: Event) => {
      if (ref.current && !ref.current.contains(e.target as Node)) cb.current()
    }
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') cb.current()
    }
    document.addEventListener('pointerdown', onDown, true)
    document.addEventListener('keydown', onEsc)
    return () => {
      document.removeEventListener('pointerdown', onDown, true)
      document.removeEventListener('keydown', onEsc)
    }
  }, [ref, enabled])
}
