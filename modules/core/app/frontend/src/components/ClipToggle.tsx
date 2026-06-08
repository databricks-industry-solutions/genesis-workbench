// Shared "add to clipboard" toggle — the compact +/✓ control used in result
// tables (Single Cell DE study list, Guided Molecule Design top-K, …). Keeps the
// copy-to-clipboard look & feel consistent across the application.
import { useClipboard, type ClipKind } from '@/stores/clipboard'

// Mirror the store's normalization so the +/✓ state matches what's stored.
const norm = (kind: ClipKind, v: string) => (kind === 'gene' ? v.trim().toUpperCase() : v.trim())

export function ClipToggle({
  kind,
  value,
  source,
  addTitle = 'Add to clipboard',
  removeTitle = 'In clipboard — click to remove',
}: {
  kind: ClipKind
  value: string
  source?: string
  addTitle?: string
  removeTitle?: string
}) {
  const toggle = useClipboard((s) => s.toggle)
  const items = useClipboard((s) => s.items)
  const inClip = items.some((i) => i.kind === kind && i.value === norm(kind, value))

  return (
    <button
      type="button"
      onClick={() => toggle({ kind, value, source })}
      title={inClip ? removeTitle : addTitle}
      className={
        'rounded border px-1.5 text-xs leading-5 ' +
        (inClip
          ? 'border-primary bg-primary/10 text-primary'
          : 'border-border text-muted-foreground hover:border-primary hover:text-primary')
      }
    >
      {inClip ? '✓' : '+'}
    </button>
  )
}
