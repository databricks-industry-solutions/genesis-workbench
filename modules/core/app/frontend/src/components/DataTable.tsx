import { flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table'
import type { ColumnDef } from '@tanstack/react-table'

import { cn } from '@/lib/utils'

/** Per-column styling hints — set via `meta` on a ColumnDef:
 *
 *   { id: 'description', header: 'Description', accessorKey: 'description',
 *     meta: { thClass: 'min-w-[260px]', tdClass: 'whitespace-normal' } }
 */
type ColumnMeta = { thClass?: string; tdClass?: string }

type DataTableProps<TData> = {
  columns: ColumnDef<TData, unknown>[]
  data: TData[]
  emptyText?: string
}

export function DataTable<TData>({ columns, data, emptyText = 'No rows' }: DataTableProps<TData>) {
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  })

  if (data.length === 0) {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
        {emptyText}
      </div>
    )
  }

  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted/50 text-xs uppercase text-muted-foreground">
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id}>
              {hg.headers.map((h) => {
                const meta = h.column.columnDef.meta as ColumnMeta | undefined
                return (
                  <th
                    key={h.id}
                    className={cn('px-3 py-2 text-left font-medium', meta?.thClass)}
                  >
                    {h.isPlaceholder
                      ? null
                      : flexRender(h.column.columnDef.header, h.getContext())}
                  </th>
                )
              })}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className="border-t border-border hover:bg-accent/20">
              {row.getVisibleCells().map((cell) => {
                const meta = cell.column.columnDef.meta as ColumnMeta | undefined
                return (
                  <td
                    key={cell.id}
                    className={cn('px-3 py-2 align-top', meta?.tdClass)}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
