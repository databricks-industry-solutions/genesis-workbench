# Genesis Workbench — Feature Documentation

This directory holds **user-facing reference documentation** for every feature exposed by the Genesis Workbench app. Each new feature shipped under `modules/core/app/views/...` must be accompanied by a doc page here.

## What belongs here

One markdown file per feature, named `<module>_<feature>.md` (snake_case). Examples:

| File | Covers |
|---|---|
| `single_cell_teddy_annotation.md` | TEDDY-based joint cell-type + disease annotation on the UMAP tab |
| `protein_studies_enzyme_optimization.md` | Guided enzyme optimization (Fast / Accurate paths) |
| `disease_biology_variant_annotation.md` | Variant annotation batch workflow |

## Required sections (template)

```markdown
# <Feature name>

**Module:** single_cell | protein_studies | small_molecule | disease_biology | …
**Status:** GA | Beta | Experimental
**Added:** YYYY-MM-DD

## What it does
One paragraph. What problem does this solve for the user? What's the input and output?

## How to use (UI walkthrough)
Step-by-step from the app: which tab, which form fields, expected wait time, where results appear.

## Inputs
Concrete schema — file formats, column names, parameter ranges. Link to example data if any.

## Outputs
What lands in MLflow (run name, artifacts, tags), what lands in Delta tables, what's shown in the result dialog.

## Underlying models / endpoints
Which serving endpoints + UC models + VS indexes the feature depends on. Link to the submodule README.

## Limitations and known issues
Cell counts, sequence lengths, accuracy caveats, currently-unsupported organisms, etc.
```

## Linking from the root README

When the feature ships, also add a bullet under the matching module in the root [`README.md`](../../../../README.md#inside-genesis-workbench)'s "Inside Genesis Workbench" section, linking to the doc page here.
