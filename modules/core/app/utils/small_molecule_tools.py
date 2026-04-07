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

EXAMPLE_PDB = """ATOM      1  N   GLN A   1     -17.226 -15.172  28.982  1.00 38.93           N
ATOM      2  CA  GLN A   1     -17.260 -14.625  27.604  1.00 32.79           C
ATOM      3  C   GLN A   1     -15.882 -14.067  27.242  1.00 24.71           C
ATOM      4  O   GLN A   1     -15.808 -13.217  26.371  1.00 25.47           O
ATOM      5  CB  GLN A   1     -18.337 -13.541  27.483  1.00 35.07           C
ATOM      6  CG  GLN A   1     -18.631 -12.751  28.769  1.00 42.43           C
ATOM      7  CD  GLN A   1     -17.438 -12.283  29.581  1.00 48.26           C
ATOM      8  OE1 GLN A   1     -16.675 -13.077  30.128  1.00 56.79           O
ATOM      9  NE2 GLN A   1     -17.292 -10.974  29.727  1.00 52.41           N
ATOM     10  N   VAL A   2     -14.824 -14.538  27.916  1.00 23.25           N
ATOM     11  CA  VAL A   2     -13.459 -14.253  27.488  1.00 20.98           C
ATOM     12  C   VAL A   2     -13.231 -14.992  26.176  1.00 20.26           C
ATOM     13  O   VAL A   2     -13.425 -16.208  26.090  1.00 21.47           O
ATOM     14  CB  VAL A   2     -12.415 -14.659  28.537  1.00 22.35           C
ATOM     15  CG1 VAL A   2     -10.993 -14.526  28.001  1.00 23.09           C
ATOM     16  CG2 VAL A   2     -12.598 -16.110  28.982  1.00 24.09           C
ATOM     17  N   GLN A   3     -12.845 -14.277  25.116  1.00 17.94           N
ATOM     18  CA  GLN A   3     -12.571 -14.860  23.803  1.00 18.21           C
ATOM     19  C   GLN A   3     -11.070 -14.898  23.557  1.00 17.63           C
ATOM     20  O   GLN A   3     -10.345 -13.948  23.845  1.00 17.87           O
ATOM     21  CB  GLN A   3     -13.241 -14.075  22.679  1.00 19.89           C
ATOM     22  CG  GLN A   3     -14.761 -14.137  22.730  1.00 22.97           C
ATOM     23  CD  GLN A   3     -15.362 -13.232  21.677  1.00 22.97           C
ATOM     24  OE1 GLN A   3     -14.770 -12.270  21.189  1.00 25.59           O
ATOM     25  NE2 GLN A   3     -16.596 -13.550  21.286  1.00 23.65           N
END
"""

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
    """Generate Mol* viewer HTML showing a protein (PDB) with a docked ligand (SDF)."""
    pdb_b64 = base64.b64encode(protein_pdb.encode()).decode()
    sdf_b64 = base64.b64encode(ligand_sdf.encode()).decode()

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

                    const pdbData = "{pdb_b64}";
                    const pdbBlob = new Blob([atob(pdbData)], {{ type: "text/plain" }});
                    const pdbUrl = URL.createObjectURL(pdbBlob);
                    await viewer.loadStructureFromUrl(pdbUrl, "pdb");

                    const sdfData = "{sdf_b64}";
                    const sdfBlob = new Blob([atob(sdfData)], {{ type: "text/plain" }});
                    const sdfUrl = URL.createObjectURL(sdfBlob);
                    await viewer.loadStructureFromUrl(sdfUrl, "sdf");
                }})();
            </script>
        </body>
    </html>"""

    return f"""<iframe style="width: 100%; height: 520px; border: none;" srcdoc='{html_str}'></iframe>"""


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
                    const pdbData = "{pdb_b64}";
                    const blob = new Blob([atob(pdbData)], {{ type: "text/plain" }});
                    const url = URL.createObjectURL(blob);
                    await viewer.loadStructureFromUrl(url, "pdb");
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
                    const pdb_{i} = "{pdb_b64}";
                    const blob_{i} = new Blob([atob(pdb_{i})], {{ type: "text/plain" }});
                    const url_{i} = URL.createObjectURL(blob_{i});
                    await viewer.loadStructureFromUrl(url_{i}, "pdb");"""
    html_str += """
                })();
            </script>
        </body>
    </html>"""
    return f"""<iframe style="width: 100%; height: 520px; border: none;" srcdoc='{html_str}'></iframe>"""
