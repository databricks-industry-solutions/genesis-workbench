// The two "bring a target sequence in" controls used across Large Molecule
// inputs — resolve from a gene (type / Clipboard / prior run) and paste a
// sequence straight off the Clipboard. The workbench idea: a target collected
// anywhere can flow into any sequence field.
import { ClipboardPaste } from '@/components/ClipboardPaste'
import { GeneResolveInput } from '@/components/GeneResolveInput'

export function SequenceSourceControls({
  onSequence,
  className = '',
}: {
  onSequence: (sequence: string) => void
  className?: string
}) {
  return (
    <div className={`flex items-center gap-1.5 ${className}`}>
      <GeneResolveInput onResolved={onSequence} />
      <ClipboardPaste kind="sequence" label="Paste sequence" onPick={(it) => onSequence(it.value)} />
    </div>
  )
}
