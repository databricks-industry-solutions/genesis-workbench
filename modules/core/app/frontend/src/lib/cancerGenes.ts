// Curated oncology gene set used to flag differential-expression hits that are
// known cancer-associated genes: canonical drivers from the COSMIC Cancer Gene
// Census plus established high-grade serous ovarian cancer (HGSOC) markers and
// proliferation/EMT hallmark genes.
//
// This is a *presentation aid* — a "★ cancer" badge, a highlighted volcano
// trace, and an optional "cancer-associated first" sort — NOT a statistical
// result. The objective signal is the combined DE score (effect × significance);
// this overlay just helps the eye land on the biologically meaningful hits.
export const CANCER_GENES: ReadonlySet<string> = new Set([
  // HGSOC / ovarian-cancer markers
  'WT1', 'PAX8', 'MUC16', 'VTCN1', 'FOLR1', 'MSLN', 'WFDC2', 'CRABP2', 'SULF1',
  'LGALS1', 'TFPI2', 'SPON1', 'CD24', 'MMP7', 'SLPI', 'CLDN3', 'CLDN4', 'KLK6',
  'KLK7', 'SOX17', 'C3',
  // Canonical drivers / oncogenes / tumour suppressors (COSMIC CGC)
  'TP53', 'BRCA1', 'BRCA2', 'RAD51', 'RAD51C', 'RAD51D', 'PALB2', 'BARD1',
  'BRIP1', 'ATM', 'ATR', 'CHEK1', 'CHEK2', 'PARP1', 'MYC', 'MYCN', 'KRAS',
  'NRAS', 'HRAS', 'EGFR', 'ERBB2', 'ERBB3', 'PIK3CA', 'AKT1', 'AKT2', 'PTEN',
  'RB1', 'CCNE1', 'CCND1', 'CDKN2A', 'CDK4', 'CDK6', 'MDM2', 'NF1', 'CTNNB1',
  'APC', 'SMAD4', 'VHL', 'MET', 'ARID1A', 'SMARCA4', 'KMT2C', 'KMT2D',
  'NOTCH1', 'FBXW7', 'BCL2', 'MCL1',
  // Proliferation / cell-cycle (malignancy hallmarks)
  'MKI67', 'TOP2A', 'PCNA', 'BIRC5', 'AURKA', 'AURKB', 'FOXM1', 'E2F1',
  'CCNB1', 'CCNB2', 'UBE2C', 'CDC20', 'CENPF', 'BUB1', 'PLK1', 'TYMS', 'RRM2',
  // EMT / invasion / metastasis
  'VIM', 'SNAI1', 'SNAI2', 'ZEB1', 'ZEB2', 'TWIST1', 'MMP2', 'MMP9', 'SPARC',
  'FN1', 'CDH2', 'S100A4',
])

/** Case-insensitive membership test for a gene symbol. */
export function isCancerGene(gene: string): boolean {
  return CANCER_GENES.has(gene.toUpperCase())
}
