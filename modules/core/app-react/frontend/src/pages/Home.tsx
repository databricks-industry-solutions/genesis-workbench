import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { Tabs } from '@/components/Tabs'
import { useUserStore, selectIsSetupDone } from '@/stores/user'
import { cn } from '@/lib/utils'

const EXAMPLE_QUESTIONS = [
  'Run a GWAS analysis',
  'Predict a protein structure',
  'Dock a molecule to a protein',
  'Annotate genetic variants',
  'Design a protein binder',
  'Analyze single cell data',
]

function AssistantTab() {
  const [query, setQuery] = useState('')
  const [submitted, setSubmitted] = useState<string | null>(null)

  const ask = useMutation({
    mutationFn: api.assistantQuery,
  })

  const runQuery = (q: string) => {
    const trimmed = q.trim()
    if (!trimmed) return
    setSubmitted(trimmed)
    ask.mutate(trimmed)
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Ask me how to do anything in this application.
      </p>
      <form
        onSubmit={(e) => {
          e.preventDefault()
          runQuery(query)
        }}
        className="flex gap-2"
      >
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Type your question…"
          className="flex-1 rounded-md border border-border bg-background px-3 py-2 text-sm"
        />
        <button
          type="submit"
          disabled={ask.isPending || !query.trim()}
          className="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:opacity-90 disabled:opacity-50"
        >
          {ask.isPending ? 'Thinking…' : 'Ask'}
        </button>
      </form>

      <div className="flex flex-wrap gap-2">
        {EXAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => {
              setQuery(q)
              runQuery(q)
            }}
            className="rounded-full border border-border px-3 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            {q}
          </button>
        ))}
      </div>

      {submitted && (
        <div className="mt-4 rounded-md border border-border bg-card p-4 text-sm">
          {ask.isPending && (
            <div className="text-muted-foreground">Thinking about &ldquo;{submitted}&rdquo;…</div>
          )}
          {ask.error && (
            <div className="text-destructive">
              I'm not able to process your request right now: {String(ask.error)}
            </div>
          )}
          {ask.data && <div className="whitespace-pre-wrap">{ask.data.answer}</div>}
        </div>
      )}
    </div>
  )
}

function SearchTab() {
  const [q, setQ] = useState('')
  const docs = useQuery({ queryKey: ['docs'], queryFn: api.docs })

  const results = useMemo(() => {
    if (!docs.data) return []
    const words = q.trim().toLowerCase().split(/\s+/).filter(Boolean)
    if (words.length === 0) return []
    return docs.data.docs.filter((d) =>
      words.every(
        (w) => d.title.toLowerCase().includes(w) || d.content.toLowerCase().includes(w),
      ),
    )
  }, [docs.data, q])

  return (
    <div className="space-y-4">
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Search workflows, methods, inputs…"
        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
      />
      {q.trim() === '' ? (
        <p className="text-xs text-muted-foreground">
          {docs.data ? `${docs.data.docs.length} workflow documents available. Type to search.` : 'Loading…'}
        </p>
      ) : results.length === 0 ? (
        <p className="text-xs text-muted-foreground">No results found.</p>
      ) : (
        <ul className="space-y-2">
          {results.map((d) => (
            <li key={d.file} className="rounded-md border border-border bg-card">
              <details>
                <summary className="cursor-pointer px-4 py-2 text-sm font-medium hover:bg-accent/30">
                  {d.title}
                </summary>
                <pre className="overflow-x-auto whitespace-pre-wrap border-t border-border bg-muted/30 px-4 py-3 text-xs">
                  {d.content}
                </pre>
              </details>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export function HomePage() {
  const bootstrap = useUserStore((s) => s.bootstrap)
  const setupDone = useUserStore(selectIsSetupDone)

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <header>
        <h1 className="text-2xl font-semibold">Home</h1>
      </header>

      {bootstrap && !setupDone && (
        <div className={cn('rounded-md border p-4 text-sm', 'border-destructive/40 bg-destructive/10')}>
          <div className="font-semibold text-destructive">⚠️ Profile setup is incomplete.</div>
          <div className="mt-1 text-muted-foreground">
            Finish your setup on the{' '}
            <Link to="/profile" className="text-primary hover:underline">
              Profile
            </Link>{' '}
            page.
          </div>
        </div>
      )}

      <Tabs
        tabs={[
          { id: 'assistant', label: 'AI Assistant', content: <AssistantTab /> },
          { id: 'search', label: 'Search Documentation', content: <SearchTab /> },
        ]}
      />
    </div>
  )
}
