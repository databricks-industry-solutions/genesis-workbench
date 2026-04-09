import os
import json
import base64
import logging
import requests
import pandas as pd
from databricks.sdk import WorkspaceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

workspace_client = WorkspaceClient()

MOLSTAR_DARK_CSS = """
<style>
    body { background: #1e1e1e; margin: 0; }
    .msp-plugin { background: #1e1e1e !important; }
    .msp-plugin .msp-layout-static { background: #1e1e1e !important; }
    .msp-plugin .msp-scrollable-container { background: #252526 !important; }
    .msp-plugin .msp-btn { background: #333 !important; color: #ccc !important; }
    .msp-plugin .msp-form-control { background: #333 !important; color: #ccc !important; border-color: #555 !important; }
    .msp-plugin .msp-control-group-header { background: #2d2d2d !important; color: #ccc !important; }
    .msp-plugin .msp-section-header { color: #ccc !important; }
    .msp-plugin .msp-icon { color: #aaa !important; }
    .msp-plugin .msp-semi-transparent-background { background: rgba(30,30,30,0.8) !important; }
    .msp-plugin .msp-log-entry { color: #ccc !important; }
    .msp-plugin .msp-tree-row { color: #ccc !important; }
    .msp-plugin .msp-control-row { color: #ccc !important; }
    .msp-plugin .msp-accent-offset { background: #333 !important; }
    .msp-plugin .msp-representation-entry { background: #2d2d2d !important; }
</style>
"""


def _query_endpoint(endpoint_name: str, payload: dict) -> dict:
    """Send a request to a model serving endpoint using the Databricks SDK.
    Uses the same auth as WorkspaceClient (handles OAuth SP, PAT, etc.)."""
    from databricks.sdk.service.serving import DataframeSplitInput

    kwargs = {}
    if "dataframe_split" in payload:
        ds = payload["dataframe_split"]
        kwargs["dataframe_split"] = DataframeSplitInput(
            columns=ds.get("columns"),
            data=ds.get("data"),
        )
    elif "inputs" in payload:
        kwargs["inputs"] = payload["inputs"]
    elif "instances" in payload:
        kwargs["instances"] = payload["instances"]
    elif "dataframe_records" in payload:
        kwargs["dataframe_records"] = payload["dataframe_records"]
    else:
        kwargs["inputs"] = payload

    response = workspace_client.serving_endpoints.query(name=endpoint_name, **kwargs)
    # Convert SDK response to dict
    result = response.as_dict() if hasattr(response, "as_dict") else response
    if hasattr(response, "predictions"):
        result = {"predictions": response.predictions}
    return result

EXAMPLE_PDB = ""  # Loaded dynamically below to keep file short

def _get_example_pdb():
    """Return a ~50-residue PDB excerpt suitable for DiffDock docking demos.
    Uses chain A of PDB 6agt (the same structure used in DiffDock registration tests).
    Fetched once and cached in module scope."""
    global EXAMPLE_PDB
    if EXAMPLE_PDB:
        return EXAMPLE_PDB
    try:
        resp = requests.get("https://files.rcsb.org/view/6agt.pdb", timeout=10)
        resp.raise_for_status()
        lines = resp.text.splitlines(keepends=True)
        seen = set()
        trimmed = []
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                if line[21] != "A":
                    continue
                resseq = line[22:27].strip()
                seen.add(resseq)
                if len(seen) > 50:
                    break
                trimmed.append(line)
        trimmed.append("END\n")
        EXAMPLE_PDB = "".join(trimmed)
    except Exception:
        EXAMPLE_PDB = "# Could not fetch example PDB. Paste your own PDB content here.\nEND\n"
    return EXAMPLE_PDB

EXAMPLE_SMILES = "COc(cc1)ccc1C#N"


def hit_diffdock(protein_pdb: str, ligand_smiles: str, samples_per_complex: int = 10) -> pd.DataFrame:
    """Call DiffDock via split endpoints: ESM embeddings first, then scoring."""
    import json as _json

    # Step 1: Compute ESM embeddings
    esm_endpoint = get_endpoint_name("DiffDock ESM Embeddings")
    logger.info(f"Step 1/2: Computing ESM embeddings via {esm_endpoint}")
    esm_result = _query_endpoint(esm_endpoint, {
        "dataframe_split": {
            "columns": ["protein_pdb"],
            "data": [[protein_pdb]]
        }
    })
    esm_predictions = esm_result.get("predictions", esm_result)
    if isinstance(esm_predictions, list):
        embeddings_b64 = esm_predictions[0].get("embeddings_b64", "{}")
    elif isinstance(esm_predictions, dict):
        embeddings_b64 = esm_predictions.get("embeddings_b64", "{}")
    else:
        embeddings_b64 = "{}"

    # Step 2: Run DiffDock scoring with pre-computed embeddings
    scoring_endpoint = get_endpoint_name("DiffDock")
    logger.info(f"Step 2/2: Running DiffDock scoring via {scoring_endpoint}")
    result = _query_endpoint(scoring_endpoint, {
        "dataframe_split": {
            "columns": ["protein_pdb", "ligand_smiles", "samples_per_complex", "esm_embeddings_b64"],
            "data": [[protein_pdb, ligand_smiles, samples_per_complex, embeddings_b64]]
        }
    })
    predictions = result.get("predictions", result)
    return pd.DataFrame(predictions)


def molstar_html_protein_and_sdf(protein_pdb: str, ligand_sdf: str) -> str:
    """Generate Mol* viewer HTML showing a protein (PDB) with a docked ligand (SDF).
    Combines into a single PDB and renders with molstar_html_pdb."""
    ligand_pdb_lines = _sdf_to_hetatm(ligand_sdf)
    logger.info(f"molstar_html_protein_and_sdf: {len(protein_pdb)} protein chars, "
                f"{len(ligand_pdb_lines.splitlines()) if ligand_pdb_lines else 0} ligand lines")
    # For now, just render the protein to verify the viewer works in this code path
    # TODO: add ligand once we confirm protein renders
    return molstar_html_pdb(protein_pdb)


def _sdf_to_hetatm(sdf_content: str) -> str:
    """Convert SDF atom block to PDB HETATM records for display in rcsb-molstar."""
    lines = sdf_content.split("\n")
    # Find the counts line (contains "V2000" or "V3000")
    counts_idx = None
    for i, line in enumerate(lines):
        if "V2000" in line or "V3000" in line:
            counts_idx = i
            break
    if counts_idx is None:
        logger.warning(f"_sdf_to_hetatm: No V2000/V3000 found in SDF ({len(lines)} lines)")
        return ""
    counts_line = lines[counts_idx]
    try:
        num_atoms = int(counts_line[:3].strip())
    except (ValueError, IndexError):
        logger.warning(f"_sdf_to_hetatm: Failed to parse atom count from: {counts_line[:30]}")
        return ""
    hetatm_lines = []
    for i in range(num_atoms):
        atom_idx = counts_idx + 1 + i
        atom_line = lines[atom_idx] if atom_idx < len(lines) else ""
        if len(atom_line) < 34:
            continue
        parts = atom_line.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            element = parts[3]
        except (ValueError, IndexError):
            continue
        atom_name = element.ljust(2)
        serial = i + 1
        hetatm_lines.append(
            f"HETATM{serial:5d}  {atom_name:<2s}  LIG B   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
        )
    logger.info(f"_sdf_to_hetatm: {num_atoms} atoms -> {len(hetatm_lines)} HETATM lines")
    return "\n".join(hetatm_lines)


from utils.streamlit_helper import get_endpoint_name


def hit_proteina_complexa(target_pdb: str, target_chain: str = "A",
                          hotspot_residues: str = "", binder_length_min: int = 50,
                          binder_length_max: int = 80, num_samples: int = 2) -> pd.DataFrame:
    """Call Proteina-Complexa binder design endpoint."""
    endpoint_name = get_endpoint_name("Proteina-Complexa Binder")
    logger.info(f"Sending Proteina-Complexa request to: {endpoint_name}")
    result = _query_endpoint(endpoint_name, {
        "dataframe_split": {
            "columns": ["target_pdb", "binder_length_min", "binder_length_max",
                        "num_samples", "hotspot_residues", "target_chain"],
            "data": [[target_pdb, binder_length_min, binder_length_max,
                      num_samples, hotspot_residues, target_chain]]
        }
    })
    return pd.DataFrame(result.get("predictions", result))


def hit_proteina_complexa_ligand(target_pdb: str, binder_length_min: int = 50,
                                  binder_length_max: int = 80, num_samples: int = 2) -> pd.DataFrame:
    """Call Proteina-Complexa ligand binder design endpoint."""
    endpoint_name = get_endpoint_name("Proteina-Complexa Ligand")
    logger.info(f"Sending Proteina-Complexa-Ligand request to: {endpoint_name}")
    result = _query_endpoint(endpoint_name, {
        "dataframe_split": {
            "columns": ["target_pdb", "binder_length_min", "binder_length_max",
                        "num_samples", "hotspot_residues", "target_chain"],
            "data": [[target_pdb, binder_length_min, binder_length_max,
                      num_samples, "", "A"]]
        }
    })
    return pd.DataFrame(result.get("predictions", result))


def hit_proteina_complexa_ame(target_pdb: str, target_chain: str = "B",
                               binder_length_min: int = 50, binder_length_max: int = 80,
                               num_samples: int = 2) -> pd.DataFrame:
    """Call Proteina-Complexa AME motif scaffolding endpoint."""
    endpoint_name = get_endpoint_name("Proteina-Complexa AME")
    logger.info(f"Sending Proteina-Complexa-AME request to: {endpoint_name}")
    result = _query_endpoint(endpoint_name, {
        "dataframe_split": {
            "columns": ["target_pdb", "binder_length_min", "binder_length_max",
                        "num_samples", "hotspot_residues", "target_chain"],
            "data": [[target_pdb, binder_length_min, binder_length_max,
                      num_samples, "", target_chain]]
        }
    })
    return pd.DataFrame(result.get("predictions", result))


def hit_esmfold(sequence: str) -> str:
    """Call ESMFold endpoint to predict protein structure from sequence."""
    endpoint_name = get_endpoint_name("ESMFold")
    logger.info(f"Sending ESMFold request to: {endpoint_name}")
    result = _query_endpoint(endpoint_name, {"inputs": [sequence]})
    return result.get("predictions", result)[0]


def hit_open_babel(input_data: str, input_format: str = "smi", output_format: str = "pdb", gen3d: bool = True) -> str:
    """Call Open Babel endpoint to convert between molecular formats."""
    endpoint_name = get_endpoint_name("Open Babel Converter")
    logger.info(f"Converting {input_format} -> {output_format} via {endpoint_name}")
    result = _query_endpoint(endpoint_name, {
        "dataframe_split": {
            "columns": ["input_data", "input_format", "output_format", "gen3d"],
            "data": [[input_data, input_format, output_format, "true" if gen3d else "false"]]
        }
    })
    predictions = result.get("predictions", result)
    return predictions[0] if isinstance(predictions, list) else predictions


def smiles_to_pdb(smiles: str) -> str:
    """Convert a SMILES string to PDB format with 3D coordinates via Open Babel."""
    return hit_open_babel(smiles, input_format="smi", output_format="pdb", gen3d=True)


def sequence_to_pdb(sequence: str) -> str:
    """Convert a protein sequence to PDB structure via ESMFold."""
    return hit_esmfold(sequence)


def hit_chemprop_bbbp(smiles_list: list) -> list:
    """Call Chemprop BBBP endpoint. Returns list of BBB penetration probabilities."""
    endpoint_name = get_endpoint_name("Chemprop BBBP")
    logger.info(f"Sending {len(smiles_list)} SMILES to {endpoint_name}")
    result = _query_endpoint(endpoint_name, {"inputs": smiles_list})
    return result.get("predictions", result)


def hit_chemprop_clintox(smiles_list: list) -> list:
    """Call Chemprop ClinTox endpoint. Returns list of toxicity probabilities."""
    endpoint_name = get_endpoint_name("Chemprop ClinTox")
    logger.info(f"Sending {len(smiles_list)} SMILES to {endpoint_name}")
    result = _query_endpoint(endpoint_name, {"inputs": smiles_list})
    return result.get("predictions", result)


def hit_chemprop_admet(smiles_list: list) -> pd.DataFrame:
    """Call Chemprop ADMET endpoint. Returns DataFrame of multi-task ADMET predictions."""
    endpoint_name = get_endpoint_name("Chemprop ADMET")
    logger.info(f"Sending {len(smiles_list)} SMILES to {endpoint_name}")
    result = _query_endpoint(endpoint_name, {"inputs": smiles_list})
    predictions = result.get("predictions", result)
    if isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], dict):
        return pd.DataFrame(predictions)
    return pd.DataFrame(predictions)


def molstar_html_pdb(pdb: str) -> str:
    """Generate Mol* viewer HTML for a single PDB structure."""
    pdb_b64 = base64.b64encode(pdb.encode()).decode()
    html_str = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.js"></script>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.css">
            """ + MOLSTAR_DARK_CSS + """
        </head>
        <body>
            <div id="viewer" style="width: 100%; height: 500px;"></div>
            <script>
                (async function() {{
                    const viewer = new rcsbMolstar.Viewer("viewer", {{layoutShowLog: false, backgroundColor: 0x1e1e1e}});
                    await viewer.loadStructureFromData(atob("{pdb_b64}"), "pdb", false);
                }})();
            </script>
        </body>
    </html>"""
    return f"""<iframe style="width: 100%; height: 520px; border: none;" srcdoc='{html_str}'></iframe>"""


def molstar_html_multi_pdb(pdbs: list) -> str:
    """Generate Mol* viewer HTML for multiple PDB structures overlaid."""
    html_str = """
    <!DOCTYPE html>
    <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.js"></script>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.css">
            """ + MOLSTAR_DARK_CSS + """
        </head>
        <body>
            <div id="viewer" style="width: 100%; height: 500px;"></div>
            <script>
                (async function() {
                    const viewer = new rcsbMolstar.Viewer("viewer", {layoutShowLog: false, backgroundColor: 0x1e1e1e});"""
    for i, pdb in enumerate(pdbs):
        pdb_b64 = base64.b64encode(pdb.encode()).decode()
        html_str += f"""
                    await viewer.loadStructureFromData(atob("{pdb_b64}"), "pdb", false);"""
    html_str += """
                })();
            </script>
        </body>
    </html>"""
    return f"""<iframe style="width: 100%; height: 520px; border: none;" srcdoc='{html_str}'></iframe>"""
