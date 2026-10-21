// Shared "job dispatched" confirmation, used by every batch-workflow tab that
// dispatches a job and tracks it under Search Past Runs (Guided Enzyme Design,
// Guided Molecule Design, RFD4 fine-tune, …). Keeps the post-start message
// consistent app-wide: a green banner with the run name + run ID.
export function DispatchSuccess({
  jobRunId,
  runUrl,
  runName,
}: {
  jobRunId: number | string
  runUrl?: string
  runName?: string
}) {
  return (
    <div className="rounded-md border border-success/40 bg-success/10 p-3 text-xs">
      <span className="text-success">✓ Job dispatched.</span>{' '}
      {runName ? (
        <>
          Run <code className="rounded bg-muted px-1">{runName}</code> (ID{' '}
          <code className="rounded bg-muted px-1">{jobRunId}</code>){' '}
        </>
      ) : (
        <>
          Run ID <code className="rounded bg-muted px-1">{jobRunId}</code>{' '}
        </>
      )}
      {runUrl && (
        <a className="text-primary hover:underline" href={runUrl} target="_blank" rel="noreferrer">
          View in Databricks ↗
        </a>
      )}
      . Track progress under <strong>Search Past Runs</strong> below.
    </div>
  )
}
