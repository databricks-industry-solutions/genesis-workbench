import os
import requests
import json
import logging
import mlflow
from databricks.sdk import WorkspaceClient

from Bio.PDB import PDBList
from Bio.PDB import PDBParser
from Bio import PDB
import tempfile

from .structure_utils import select_and_align

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
workspace_client = WorkspaceClient()

def hit_model_endpoint(endpoint_name, inputs) -> str:
    """
    Query endpoint with input
    """

    endpoint_name = f"gwb_{endpoint_name}_endpoint"

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
    return hit_model_endpoint('esmfold', [sequence])[0]

@mlflow.trace(span_type="TOOL")
def hit_rfdiffusion(input_dict):
    return hit_model_endpoint('rfdiffusion_inpainting', [input_dict])[0]

@mlflow.trace(span_type="TOOL")
def hit_proteinmpnn(pdb_str):
    return hit_model_endpoint('proteinmpnn', [pdb_str])

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
def make_designs(sequence, n_rfdiffusion_hits=1, progress_callback=lambda x: print(x)):
    #callback arguments: {status_parsing: aa, status_esm_init: bb, status_rfdiffusion: cc, status_proteinmpnn : dd,status_esm_preds:ee}
    #
    progress_report = {
        "status_parsing": 0, 
        "status_esm_init": 0, 
        "status_rfdiffusion": 0, 
        "status_proteinmpnn" : 0,
        "status_esm_preds":0
    }

    seq_details = parse_sequence(sequence)

    logger.debug("======Seq details:")
    logger.debug(seq_details)
    logger.debug("======")
    progress_report["status_parsing"] = 100
    progress_callback(progress_report)
    
    esmfold_initial = hit_esmfold(seq_details['sequence'])

    logger.debug("======esmfold_initial:")
    logger.debug(esmfold_initial)
    logger.debug("======")
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

    logger.debug("======modified_pdb_text:")
    logger.debug(modified_pdb_text)
    logger.debug("======")

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



    all_seqs = []
    n=1
    for pdb_ in designed_pdb_strs:
        all_seqs.extend(hit_proteinmpnn(pdb_))

        progress_report["status_proteinmpnn"] = int( ( n/ len(designed_pdb_strs) ) * 100 )
        progress_callback(progress_report)
        n+=1
    
    all_pdbs = []
    n=1
    for s in all_seqs:
        all_pdbs.append(hit_esmfold(s))

        progress_report["status_esm_preds"] = int( ( n/ len(all_seqs) ) * 100 )
        progress_callback(progress_report)
        n+=1

    return {
        'initial': esmfold_initial,
        'designed': all_pdbs
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

