import { useEffect, useRef, type RefObject } from 'react'

type AnyRef = RefObject<HTMLElement | null>

/**
 * Dismiss a popover/dropdown on an outside click or the Escape key.
 *
 * Single source of truth for "click-away to close" — use this for every
 * lightweight popover instead of hand-rolling a document listener per component
 * (modals use Dialog/Drawer instead).
 *
 * Pass the actual interactive elements that count as "inside" — typically the
 * trigger BUTTON and the PANEL — NOT a wrapper div. A wrapper can be
 * unexpectedly large (an inline-block/flex container can stretch to its row),
 * and `wrapper.contains(target)` would then be true for clicks far away, so the
 * popover never closes. Anchoring to the button + panel (both content-sized)
 * makes the check immune to the wrapper's size.
 *
 * Implementation notes:
 *  - CAPTURE-phase `pointerdown`: fires before any ancestor that calls
 *    `stopPropagation()` on a bubble-phase mousedown could swallow it.
 *  - The trigger button is "inside", so its own onClick toggles cleanly and the
 *    close doesn't fire first (which would otherwise let the click reopen it).
 *  - Listeners attach only while `enabled` (the popover is open).
 *  - `onClose` is read through a ref so an inline arrow doesn't re-subscribe.
 *
 * @param insideRefs ref or refs to the element(s) that count as inside —
 *                   e.g. [triggerButtonRef, panelRef]
 * @param onClose    invoked on an outside click or Escape
 * @param enabled    whether the popover is currently open
 */
export function useOutsideDismiss(
  insideRefs: AnyRef | AnyRef[],
  onClose: () => void,
  enabled: boolean,
): void {
  const cb = useRef(onClose)
  cb.current = onClose
  const refsRef = useRef(insideRefs)
  refsRef.current = insideRefs

  useEffect(() => {
    if (!enabled) return
    const isInside = (target: Node) => {
      const refs = Array.isArray(refsRef.current) ? refsRef.current : [refsRef.current]
      return refs.some((r) => r.current && r.current.contains(target))
    }
    const onDown = (e: Event) => {
      if (!isInside(e.target as Node)) cb.current()
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
  }, [enabled])
}
