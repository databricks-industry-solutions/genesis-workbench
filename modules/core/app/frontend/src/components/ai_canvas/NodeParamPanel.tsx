// Vortex (ai_canvas) — right panel to inspect/edit the selected node's params.
import { CATEGORY_STYLE } from './graph'
import type { VortexNode } from './graph'
import type { CanvasParam } from '@/types/api'

export function NodeParamPanel({
  node,
  workflowName,
  onChangeWorkflowName,
  onChangeParam,
  onRename,
  onDelete,
}: {
  node: VortexNode | null
  workflowName: string
  onChangeWorkflowName: (name: string) => void
  onChangeParam: (name: string, value: unknown) => void
  onRename: (label: string) => void
  onDelete: () => void
}) {
  const cat = node?.data.catalog
  const category = cat?.category ?? 'endpoint'

  return (
    <div className="flex h-full w-64 shrink-0 flex-col border-l border-border bg-card/40">
      {/* Workflow name — always visible, auto-generated on start, editable. */}
      <div className="border-b border-border p-3">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          Workflow name
        </div>
        <input
          value={workflowName}
          onChange={(e) => onChangeWorkflowName(e.target.value)}
          placeholder="vortex_…"
          className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm font-semibold"
        />
        <p className="mt-1 text-[10px] leading-snug text-muted-foreground">
          Saved to your MLflow experiment folder under <code>/ai_canvas</code>.
        </p>
      </div>

      {!node ? (
        <div className="flex flex-1 items-center justify-center p-4 text-center text-xs text-muted-foreground">
          Select a node to edit its inputs and parameters.
        </div>
      ) : (
        <>
          <div className="border-b border-border p-3">
            <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              {CATEGORY_STYLE[category].label}
            </div>
            <input
              value={node.data.label}
              onChange={(e) => onRename(e.target.value)}
              className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm font-semibold"
            />
            {cat?.description && (
              <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{cat.description}</p>
            )}
          </div>

          <div className="flex-1 space-y-3 overflow-auto p-3">
            {cat && cat.params.length === 0 && (
              <p className="text-xs text-muted-foreground">No parameters for this node.</p>
            )}
            {cat?.params.map((p) => (
              <ParamInput
                key={p.name}
                param={p}
                value={node.data.params?.[p.name]}
                onChange={(v) => onChangeParam(p.name, v)}
              />
            ))}

            {cat && (cat.inputs.length > 0 || cat.outputs.length > 0) && (
              <div className="border-t border-border pt-2 text-[11px] text-muted-foreground">
                {cat.inputs.length > 0 && (
                  <div>
                    <span className="font-medium">Inputs:</span>{' '}
                    {cat.inputs.map((p) => `${p.label} (${p.dtype})`).join(', ')}
                  </div>
                )}
                {cat.outputs.length > 0 && (
                  <div className="mt-1">
                    <span className="font-medium">Outputs:</span>{' '}
                    {cat.outputs.map((p) => `${p.label} (${p.dtype})`).join(', ')}
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="border-t border-border p-3">
            <button
              onClick={onDelete}
              className="w-full rounded-md border border-destructive/40 px-2 py-1.5 text-xs font-medium text-destructive hover:bg-destructive/10"
            >
              Delete node
            </button>
          </div>
        </>
      )}
    </div>
  )
}

function ParamInput({
  param,
  value,
  onChange,
}: {
  param: CanvasParam
  value: unknown
  onChange: (v: unknown) => void
}) {
  const label = (
    <div className="mb-1 flex items-center gap-1 text-xs font-medium text-foreground">
      {param.label}
      {param.required && <span className="text-destructive">*</span>}
    </div>
  )

  let control
  if (param.type === 'bool') {
    control = (
      <input
        type="checkbox"
        checked={Boolean(value)}
        onChange={(e) => onChange(e.target.checked)}
        className="h-4 w-4"
      />
    )
  } else if (param.type === 'select') {
    control = (
      <select
        value={String(value ?? '')}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs"
      >
        {param.options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    )
  } else if (param.type === 'text') {
    control = (
      <textarea
        value={String(value ?? '')}
        onChange={(e) => onChange(e.target.value)}
        rows={3}
        className="w-full rounded-md border border-border bg-background px-2 py-1 font-mono text-xs"
      />
    )
  } else {
    const numeric = param.type === 'int' || param.type === 'float'
    control = (
      <input
        type={numeric ? 'number' : 'text'}
        step={param.type === 'float' ? 'any' : undefined}
        value={value === undefined || value === null ? '' : String(value)}
        onChange={(e) => {
          const raw = e.target.value
          if (!numeric) return onChange(raw)
          if (raw === '') return onChange(null)
          onChange(param.type === 'int' ? parseInt(raw, 10) : parseFloat(raw))
        }}
        className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs"
      />
    )
  }

  return (
    <label className="block">
      {label}
      {control}
      {param.help && <p className="mt-1 text-[10px] text-muted-foreground">{param.help}</p>}
    </label>
  )
}
