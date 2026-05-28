export function WorkflowComingSoon({ name }: { name: string }) {
  return (
    <div className="rounded-md border border-border bg-muted/30 p-6 text-sm text-muted-foreground">
      <div className="mb-1 font-medium text-foreground">{name}</div>
      Not yet ported from Streamlit. This workflow lands in a future phase.
    </div>
  )
}
