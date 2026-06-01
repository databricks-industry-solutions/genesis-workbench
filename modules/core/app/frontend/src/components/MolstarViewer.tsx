type Props = {
  viewerHtml: string | null | undefined
  height?: number | string
  title?: string
}

export function MolstarViewer({ viewerHtml, height = 520, title = 'Structure viewer' }: Props) {
  if (!viewerHtml) {
    return (
      <div
        className="flex items-center justify-center rounded-md border border-border bg-muted/30 text-sm text-muted-foreground"
        style={{ height }}
      >
        No structure to display.
      </div>
    )
  }
  return (
    <iframe
      title={title}
      srcDoc={viewerHtml}
      sandbox="allow-scripts allow-same-origin"
      className="w-full rounded-md border border-border bg-[#1e1e1e]"
      style={{ height }}
    />
  )
}
