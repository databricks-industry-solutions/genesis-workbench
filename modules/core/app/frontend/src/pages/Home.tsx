import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import ReactMarkdown from 'react-markdown'
import type { Components } from 'react-markdown'

import { api } from '@/api/client'
import { DnaLoader } from '@/components/DnaLoader'
import { Tabs } from '@/components/Tabs'
import { useUserStore, selectIsSetupDone } from '@/stores/user'
import { cn } from '@/lib/utils'

// Custom element mapping so the rendered markdown picks up the app's
// existing Tailwind tokens (text-foreground, muted, border, etc.) without
// pulling in @tailwindcss/typography.
const markdownComponents: Components = {
  h1: ({ children }) => (
    <h1 className="mb-3 mt-4 text-xl font-semibold text-foreground">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="mb-2 mt-4 text-lg font-semibold text-foreground">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="mb-2 mt-3 text-sm font-semibold text-foreground">{children}</h3>
  ),
  h4: ({ children }) => (
    <h4 className="mb-1 mt-2 text-sm font-medium text-foreground">{children}</h4>
  ),
  p: ({ children }) => (
    <p className="mb-2 text-sm leading-relaxed text-foreground">{children}</p>
  ),
  ul: ({ children }) => (
    <ul className="mb-2 list-disc space-y-1 pl-5 text-sm">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-2 list-decimal space-y-1 pl-5 text-sm">{children}</ol>
  ),
  li: ({ children }) => <li className="text-sm leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  code: ({ children, className }) =>
    className ? (
      // fenced code blocks — react-markdown wraps these in <pre><code>
      <code className={className}>{children}</code>
    ) : (
      <code className="rounded bg-muted px-1 py-0.5 font-mono text-[12px]">{children}</code>
    ),
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-md border border-border bg-muted/30 p-3 text-xs">
      {children}
    </pre>
  ),
  a: ({ children, href }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="text-primary hover:underline"
    >
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-border pl-3 text-sm text-muted-foreground">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-3 border-border" />,
  table: ({ children }) => (
    <table className="my-2 w-full border-collapse text-xs">{children}</table>
  ),
  th: ({ children }) => (
    <th className="border border-border bg-muted/50 px-2 py-1 text-left font-medium">
      {children}
    </th>
  ),
  td: ({ children }) => <td className="border border-border px-2 py-1">{children}</td>,
}

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

  // Shares the cache with SearchTab via the ['docs'] key so the docs round-
  // trip happens once per session regardless of which tab opens first.
  const docs = useQuery({ queryKey: ['docs'], queryFn: api.docs })

  const relatedDocs = useMemo(() => {
    if (!docs.data || !ask.data || !submitted) return []
    const answer = ask.data.answer.toLowerCase()
    const queryWords = submitted
      .trim()
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 2)
    if (queryWords.length === 0) return []
    return docs.data.docs.filter((d) => {
      const t = d.title.toLowerCase()
      const c = d.content.toLowerCase()
      // Surface a doc if the assistant's answer mentions its title, or if
      // every word in the user's query appears somewhere in the doc.
      return (
        answer.includes(t) ||
        queryWords.every((w) => t.includes(w) || c.includes(w))
      )
    })
  }, [docs.data, ask.data, submitted])

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
            <DnaLoader label={`Thinking about “${submitted}”…`} />
          )}
          {ask.error && (
            <div className="text-destructive">
              I'm not able to process your request right now: {String(ask.error)}
            </div>
          )}
          {ask.data && (
            <ReactMarkdown components={markdownComponents}>
              {ask.data.answer}
            </ReactMarkdown>
          )}
        </div>
      )}

      {ask.data && relatedDocs.length > 0 && (
        <div className="mt-2 space-y-2">
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            Related documentation
          </div>
          <ul className="space-y-2">
            {relatedDocs.map((d) => (
              <li key={d.file} className="rounded-md border border-border bg-card">
                <details>
                  <summary className="cursor-pointer px-4 py-2 text-sm font-medium hover:bg-accent/30">
                    {d.title}
                  </summary>
                  <div className="border-t border-border bg-muted/10 px-4 py-3">
                    <ReactMarkdown components={markdownComponents}>{d.content}</ReactMarkdown>
                  </div>
                </details>
              </li>
            ))}
          </ul>
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
                <div className="border-t border-border bg-muted/10 px-4 py-3">
                  <ReactMarkdown components={markdownComponents}>{d.content}</ReactMarkdown>
                </div>
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
    <div className="space-y-6 px-8 py-8">
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
