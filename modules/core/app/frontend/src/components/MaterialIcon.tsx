// Renders a Material Symbols (Outlined) glyph. NOTE: the font is loaded as a
// ligature *subset* in index.html — add any new icon name to that `icon_names=`
// query param or it won't render.
import { cn } from '@/lib/utils'

export function MaterialIcon({ name, className }: { name: string; className?: string }) {
  return (
    <span aria-hidden className={cn('material-symbols-outlined leading-none', className)}>
      {name}
    </span>
  )
}
