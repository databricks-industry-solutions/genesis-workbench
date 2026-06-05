// Builds the annotated label for a cluster <select> option, shared by the
// Pathway Enrichment and Perturbation tabs:
//   "Cluster N | <SCimilarity> (conf%) | <TEDDY> (conf%) | <TEDDY disease>"
// Parts are omitted when the corresponding annotation isn't present.
import type { SavedAnnotationsResponse } from '@/types/api'

export function clusterOptionLabel(
  cluster: string,
  anno: SavedAnnotationsResponse | undefined,
): string {
  const parts = [`Cluster ${cluster}`]
  const s = anno?.scimilarity?.annotations.find((a) => a.cluster === cluster)
  const t = anno?.teddy?.annotations.find((a) => a.cluster === cluster)
  if (s) parts.push(`${s.predicted_cell_type} (${Math.round(s.confidence_pct)}%)`)
  if (t) {
    parts.push(`${t.predicted_cell_type} (${Math.round(t.cell_type_confidence_pct)}%)`)
    if (t.predicted_disease && t.predicted_disease !== 'Unknown') parts.push(t.predicted_disease)
  }
  return parts.join(' | ')
}
