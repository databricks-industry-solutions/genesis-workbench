import os
import requests
import json
import logging
import mlflow
from dataclasses import dataclass, asdict

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

from Bio.PDB import PDBList
from Bio.PDB import PDBParser
from Bio import PDB
import tempfile

from genesis_workbench.models import set_mlflow_experiment
from genesis_workbench.workbench import UserInfo

from .structure_utils import select_and_align
from .streamlit_helper import get_endpoint_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
workspace_client = WorkspaceClient()

# Boltz cold starts on GPU_SMALL routinely exceed the SDK's 60s default — same
# pattern that bit SCimilarity. Use a 600s client by default for Boltz; allow
# per-call override via hit_boltz(..., timeout_seconds=...).
_BOLTZ_DEFAULT_TIMEOUT_SECONDS = 600
_boltz_long_timeout_client = WorkspaceClient(
    config=Config(http_timeout_seconds=_BOLTZ_DEFAULT_TIMEOUT_SECONDS)
)

def hit_model_endpoint(display_name, inputs) -> str:
    """Query a model serving endpoint by display name."""
    endpoint_name = get_endpoint_name(display_name)
    try:
        logger.info(f"Sending request to model endpoint: {endpoint_name}")
        response = workspace_client.serving_endpoints.query(
            name=endpoint_name,
            inputs=inputs
        )
        logger.info("Received response from model endpoint")
        return response.predictions
    except Exception as e:
        logger.error(f"Error querying model: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

@mlflow.trace(span_type="LLM")
def hit_esmfold(sequence):
    return hit_model_endpoint('ESMFold', [sequence])[0]

@mlflow.trace(span_type="TOOL")
def hit_rfdiffusion(input_dict):
    return hit_model_endpoint('RFDiffusion', [input_dict])[0]

@mlflow.trace(span_type="LLM")
def hit_boltz(sequence, msa="no_msa", use_msa_server="True", timeout_seconds=None):
    """Call Boltz endpoint for structure prediction.

    Args:
        sequence: Protein sequence string, or a formatted input string like
                  "protein_A:SEQ;rna_B:SEQ" for multi-chain complexes.
        msa: MSA option - "no_msa" for fast prediction, or path to MSA file.
        use_msa_server: "True" to use MSA server, "False" to skip.
        timeout_seconds: Optional per-call HTTP timeout override. Defaults to
                         the module-level _BOLTZ_DEFAULT_TIMEOUT_SECONDS (600s).
                         The optimization loop passes 900s for ligand complexes.

    Returns:
        PDB string of the predicted structure.
    """
    # If the user passed a plain sequence, wrap it in the Boltz input format
    if not any(prefix in sequence for prefix in ["protein_", "rna_", "dna_", "smiles_", ":"]):
        boltz_input = f"protein_A:{sequence}"
    else:
        boltz_input = sequence

    payload = [{
        "input": boltz_input,
        "msa": msa,
        "use_msa_server": use_msa_server,
    }]

    if timeout_seconds is not None and timeout_seconds != _BOLTZ_DEFAULT_TIMEOUT_SECONDS:
        client = WorkspaceClient(config=Config(http_timeout_seconds=timeout_seconds))
    else:
        client = _boltz_long_timeout_client

    endpoint_name = get_endpoint_name("Boltz")
    try:
        logger.info(f"Sending request to Boltz endpoint: {endpoint_name}")
        response = client.serving_endpoints.query(
            name=endpoint_name,
            inputs=payload,
        )
        logger.info("Received response from Boltz endpoint")
        result = response.predictions
        if isinstance(result, list) and len(result) > 0:
            entry = result[0]
            if isinstance(entry, dict) and "pdb" in entry:
                return entry["pdb"]
            return entry
        return result
    except Exception as e:
        logger.error(f"Error querying Boltz: {str(e)}", exc_info=True)
        raise

@mlflow.trace(span_type="TOOL")
def hit_proteinmpnn(pdb_str):
    return hit_model_endpoint('ProteinMPNN', [pdb_str])

@mlflow.trace(span_type="TOOL")
def extract_chain_reindex(structure, chain_id='A'):
    # Extract chain A
    chain = structure[0][chain_id]
    
    # Create a new structure with only chain A & 1-indexed
    new_structure = PDB.Structure.Structure('new_structure')
    new_model = PDB.Model.Model(0)
    new_chain = PDB.Chain.Chain(chain_id)
    
    # Reindex residues starting from 1
    for i, residue in enumerate(chain, start=1):
        if residue.id[0] == ' ':  # Ensure no HETATM
            residue.id = (' ', i, ' ')
            new_chain.add(residue)
    
    new_model.add(new_chain)
    new_structure.add(new_model)
    
    # Save the new structure to a PDB file
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    with tempfile.NamedTemporaryFile(suffix='.pdb') as f:
        io.save(f.name)
        with open(f.name, 'r') as f_handle:
            pdb_text = f_handle.read()
    return pdb_text

@mlflow.trace(span_type="TOOL")
def parse_sequence(sequence):
    # Get index of "[" and "]"
    start_idx = sequence.find("[")
    end_idx = sequence.find("]")

    raw_sequence = sequence.replace("[", "").replace("]", "")
    return {
        "sequence": raw_sequence,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }

@mlflow.trace(span_type="TOOL")
def make_designs(sequence, 
                 mlflow_experiment_name:str,
                 mlflow_run_name:str,
                 user_info: UserInfo,
                 n_rfdiffusion_hits=1, 
                 progress_callback=lambda x: print(x) ):
    
    #get the hostname and user token
    host_name = os.environ["DATABRICKS_HOST"]

    experiment = set_mlflow_experiment(experiment_tag = mlflow_experiment_name,
                                       user_email = user_info.user_email,
                                       host = None, #host_name, Until we a have a way to grant user token permission to workspace api
                                       token = None #user_info.user_access_token
                                       )
    mlflow_run_id = 0

    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:

        mlflow_run_id = run.info.run_id
        mlflow.log_param("sequence",sequence)
        mlflow.log_param("n_rfdiffusion_hits",n_rfdiffusion_hits)

        progress_report = {
            "status_parsing": 0, 
            "status_esm_init": 0, 
            "status_rfdiffusion": 0, 
            "status_proteinmpnn" : 0,
            "status_esm_preds":0
        }

        seq_details = parse_sequence(sequence)
        mlflow.log_param("seq_details", json.dumps(seq_details))
        progress_report["status_parsing"] = 100
        progress_callback(progress_report)

        esmfold_initial = hit_esmfold(seq_details['sequence'])
        mlflow.log_dict({"predictions":esmfold_initial}, "esmfold_initial_predictions.json")

        progress_report["status_esm_init"] = 100
        progress_callback(progress_report)

        # take the output and modify so that bewteen start and end idx residues are replaced with Glycine and only CA kept
        with tempfile.NamedTemporaryFile(suffix='.pdb') as f:
            with open(f.name, 'w') as fw:
                fw.write(esmfold_initial)
            structure = PDBParser().get_structure("esmfold", f.name)
        
        modified_pdb_text = extract_chain_reindex(
            structure, 
            chain_id='A'
        )

        mlflow.log_dict({"modified_pdb_text":modified_pdb_text}, "modified_pdb_text.json")


        # now pass that modified structure to rfdifffusion as string
        designed_pdb_strs = []
        for i in range(n_rfdiffusion_hits):
            designed_pdb = hit_rfdiffusion({
                'pdb': modified_pdb_text,
                'start_idx': seq_details['start_idx'],
                'end_idx': seq_details['end_idx'],
            })
            designed_pdb_strs.append(designed_pdb)
            
            progress_report["status_rfdiffusion"] = int( ( (i+1)/n_rfdiffusion_hits) * 100 )
            progress_callback(progress_report)


        mlflow.log_dict({"designed_pdb_strs":designed_pdb_strs}, "designed_pdb_strs.json")

        all_seqs = []
        n=1
        for pdb_ in designed_pdb_strs:
            all_seqs.extend(hit_proteinmpnn(pdb_))

            progress_report["status_proteinmpnn"] = int( ( n/ len(designed_pdb_strs) ) * 100 )
            progress_callback(progress_report)
            n+=1
        
        mlflow.log_dict({"protein_mpnn_seqs":all_seqs}, "protein_mpnn_seqs.json")


        all_pdbs = []
        n=1
        for s in all_seqs:
            all_pdbs.append(hit_esmfold(s))

            progress_report["status_esm_preds"] = int( ( n/ len(all_seqs) ) * 100 )
            progress_callback(progress_report)
            n+=1

        mlflow.log_dict({"all_pdb_results":all_pdbs}, "all_pdb_results.json")

        return {
            'initial': esmfold_initial,
            'designed': all_pdbs,
            'experiment_id' : experiment.experiment_id,
            'run_id': mlflow_run_id
        }

def align_designed_pdbs(designed_pdbs):
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(len(designed_pdbs['designed'])):
            with open(os.path.join(tmpdir,f"d_{i}_structure.pdb"), 'w') as f:
                f.write(designed_pdbs['designed'][i])
        with open(os.path.join(tmpdir,"init_structure.pdb"), 'w') as f:
            f.write(designed_pdbs['initial'])

        init_structure = PDBParser().get_structure("esmfold_initial", os.path.join(tmpdir,"init_structure.pdb"))
        unaligned_structures = []
        for i in range(len(designed_pdbs['designed'])):
            unaligned_structures.append( PDBParser().get_structure("designed", os.path.join(tmpdir,f"d_{i}_structure.pdb")) )

    aligned_structures = []
    for i, ua in enumerate(unaligned_structures):
        init_structure_str, true_structure_str = select_and_align(
            init_structure, ua
        )
        if i==0:
            aligned_structures.append(init_structure_str)  
        aligned_structures.append(true_structure_str) 
    return aligned_structures