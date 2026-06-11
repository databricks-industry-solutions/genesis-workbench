# AI-Assisted Workflow Generation (Vortex)

## Introduction

Vortex is a visual, drag-and-drop canvas for composing multi-step pipelines from the building blocks Genesis Workbench already deploys — model-serving endpoints, prebuilt workflows (Guided Enzyme/Molecule Optimization, Protein Design, ADMET Screen, AlphaFold, GWAS, …), data input/output nodes, and small deterministic transforms. A pipeline is wired on the canvas and dispatched to a single orchestrator Databricks Job that runs each node in topological order and tracks the whole run in MLflow. Vortex can also **generate** a workflow from a plain-language goal: an LLM drafts the graph, then reviews and corrects its own draft before handing it back.

## What It Achieves

- Compose endpoints, prebuilt workflows, transforms, and data IO into a single governed pipeline — no code.
- Generate a complete workflow from a natural-language goal, with the model's reasoning streamed live as it designs.
- **Self-review** the generated graph: drop hallucinated/non-runnable nodes, strip type-incompatible edges, and flag pipelines that don't actually accomplish the goal (e.g. a screen that scores a raw seed instead of the optimized candidates) — then repair them.
- Validate before running: a live checklist lists every unconnected input, empty value, or bad path, and the Run button is gated until the graph is runnable.
- Track every run under **Past Runs** with a read-only result canvas (per-node passed/failed/skipped), a JSON view/copy, and one-click **Re-run**.
- **Fail loudly on bad data** — a transform that resolves to null (or any node fed a null upstream value) fails the run instead of silently producing a meaningless result.

## How to Use

Open the **Vortex** tab (Home). Either:

- **Build by hand** — drag nodes from the left palette onto the canvas, wire an output port to a compatible input port, set params on the right panel, and click **Run**. Inputs are *convertible*: type a value inline on the node, or wire it from an upstream node (an edge overrides the inline value).
- **Generate with AI** — type a goal (e.g. *"Optimize an enzyme for a substrate and validate the top design with AlphaFold"*) and click **Generate workflow**. The model streams its plan, drafts the graph, then reviews and rewires it; the finished pipeline lands on the canvas ready to edit and run.

Use **Save**/**Load** to persist workflows, **Past Runs** to browse history, and the `{ }` button (canvas top-right, and on each result) to view/copy the graph JSON.

### Inputs

- A natural-language **goal** (for AI generation), and/or hand-placed nodes.
- Per-node **inputs** (wired from upstream or typed inline) and **params**.
- Data sources: `text_input` (a literal), `volume_input` (a UC Volume path — also supplies file-backed inputs such as a PDB structure or a FASTA), `delta_input` (a `catalog.schema.table`).

### Outputs

- A dispatched orchestrator run with an MLflow run id and link; per-node status tags (`node:<id>:status`), a `graph.json` artifact, and a `workflow_results.json` artifact holding every node's output and the final `output_sink` values.
- Surfaced in **Past Runs**: status, a result canvas colored by per-node outcome, and the collected outputs.

## How It's Implemented

### Architecture

- **Canvas (frontend)** — React Flow. Nodes/edges are converted to/from a portable graph JSON; edges only connect dtype-compatible ports (`portsCompatible`), and a validation pass gates Run.
- **Catalog + registry** — the backend builds a live catalog of runnable node types (deployed endpoints, batch jobs, chains, transforms, IO) from `ai_canvas_registry.py`; this catalog is the contract the AI generator must conform to.
- **AI generation** — `services/ai_canvas.py` prompts the LLM with the catalog, parses the draft, **validates** it (drops non-offerable nodes and type-incompatible edges), then **reviews** it against the goal (semantic + structural lint → LLM repair, converging when the graph is unchanged). Progress streams to the UI as Server-Sent Events.
- **Orchestrator** — `notebooks/run_ai_canvas_workflow.py` runs on a serverless Job: it loads the enriched graph, resolves each node to an execution descriptor, runs nodes in topological order, refuses to run a node on a null wired input, and writes per-node status + results to MLflow.
- **Shared capability core** — endpoints, chains, and transforms execute through `genesis_workbench/executor.py` (also used by the MCP server), so the canvas, the orchestrator, and MCP all run the same code.

### Key Files

- `modules/core/app/frontend/src/components/ai_canvas/` — `VortexTab.tsx`, `graph.ts`, `CanvasNode.tsx`, `RunHistory.tsx`, `ResultGraph.tsx`, `GraphJson.tsx`
- `modules/core/app/backend/app/services/ai_canvas.py` and `app/routers/ai_canvas.py` — catalog, generation + self-review, run dispatch, status/results, search
- `modules/core/app/backend/app/services/ai_canvas_registry.py` — node-type registry / catalog
- `modules/core/notebooks/run_ai_canvas_workflow.py` — the orchestrator job
- `modules/core/library/genesis_workbench/src/genesis_workbench/{capabilities,executor}.py` — shared capability core (also backs MCP)
