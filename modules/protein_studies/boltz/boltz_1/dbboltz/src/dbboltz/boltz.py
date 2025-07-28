from boltz.main import predict as boltz_predict
import tempfile
import yaml
import json
import os
import numpy as np
from typing import Optional, Union, Dict, List, Tuple, Callable, Any
from rdkit import Chem
import re
import mlflow
from collections import defaultdict

from dbboltz.alphafold.parsers import (
    convert_stockholm_to_a3m,
    deduplicate_stockholm_msa,
    remove_empty_columns_from_stockholm_msa
)

from dbboltz.alphafold.pipeline import run_msa_tool
from dbboltz.alphafold import jackhmmer

INT_INPUTS = [
    'msa_depth',
    'diffusion_samples',
    'recycling_steps',
    'sampling_steps',
]

FLOAT_INPUTS = [
]

BOOL_INPUTS = [
    'use_msa_server',
]

CONFIDENCE_ENTRIES_KEEP_SERVING = [
    'confidence_score',
    'ptm',
    'iptm',
    'ligand_iptm',
    'protein_iptm',
    'complex_plddt',
    'complex_iplddt',
    'complex_pde',
    'complex_ipde',
]

DEFAULT_JH_PARAMS = {
    'jh': {},
    'no_msa': {},
    'mmseqs': {}
}

DEFAULT_PARAMS = {
    'msa': 'no_msa',
    'msa_depth': 20,
    'l2_distance_threshold':2.0,
    'diffusion_samples': 1,
    'recycling_steps': 3,
    'sampling_steps': 200,
}


def convert_sto_to_a3m(sto_path=None, sto_str=None):
    if sto_str is None:
        if sto_path is not None:
            with open(sto_path, 'r') as f:
                msa_for_templates = f.read()
        else:
            raise ValueError("Either sto_path or msa_for_templates must be provided.")
    else:
        msa_for_templates = sto_str
    
    msa_for_templates = deduplicate_stockholm_msa(msa_for_templates)
    try:
        msa_for_templates = remove_empty_columns_from_stockholm_msa(
            msa_for_templates)
    except:
        pass
    msa_as_a3m = convert_stockholm_to_a3m(msa_for_templates)
    return msa_as_a3m

@mlflow.trace(span_type='TOOL')
def get_fasta_contents(file):
    with open(file, 'r') as f:
        file_contents = f.read()
    return file_contents

@mlflow.trace(span_type='TOOL')
def get_jackhmmer_alignment(
    query: str, 
    sequences: Union[List[str], str], 
    jackhmmer_binary_path: str, 
    as_a3m: Optional[bool] =True,
    jh_kwargs: Optional[Dict]= None
    ):
    """
    Run jackhmmer on a list of sequences, or a fasta file and return the alignment as a3m

    Args:
        query (str): The query protein sequence
        sequences (Union[List[str], str]): A list of sequences or if single str a path fo a fasta file
        jackhmmer_binary_path (str): Path to jackhmmer binary
        as_a3m (Optional[bool]): If True, return the alignment as a3m, otherwise return the alignment as a fasta file
        jh_kwargs (Optional[Dict]): Keyword arguments for jackhmmer

    Returns:
        a3m_text (str): The alignment as a3m
    """
    # TODO make jackhmmer optional by using which jackhmmer?

    if jh_kwargs is None:
        jh_kwargs = dict()

    # Create temporary input and output files
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.fasta') as in_file, \
        tempfile.NamedTemporaryFile(mode='w+', suffix='.fasta') as q_file, \
        tempfile.NamedTemporaryFile(mode='w+', suffix='.sto') as out_file:

        if isinstance(sequences, list):
            for name, seq in sequences:
                if name.startswith('>'):
                    name_ = name[1:]
                else:
                    name_ = name
                in_file.write(f">{name_}\n{seq}\n")
            in_file.flush()
            infasta = get_fasta_contents(in_file.name)
            in_file_name = in_file.name
        elif isinstance(sequences, str):
            if os.path.exists(sequences):
                in_file_name = sequences
            else:
                raise ValueError("sequences must be a list of sequences or a path to a fasta file: you passed a string to a file that does not exist")
        else:
            raise ValueError("sequences must be a list of sequences or a path to a fasta file")

        jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=in_file_name,
            **jh_kwargs
        )

        input_sequence = query 
        input_description = "protein"

        q_file.write(f">{input_description}\n{input_sequence}\n")
        q_file.flush()

        out_path = out_file.name
        jackhmmer_uniref90_result = run_msa_tool(
            msa_runner=jackhmmer_uniref90_runner,
            input_fasta_path=q_file.name,
            msa_out_path=out_path,
            msa_format='sto',
            use_precomputed_msas=False,
            max_sto_sequences=100
        )
        a3m_text = convert_sto_to_a3m(out_path)
    return a3m_text

@mlflow.trace(span_type='TOOL')
def post_process_boltz_results(dir, yaml_name, expected_result_count : int =0):
    preds_dir = f"{dir}/boltz_results_{yaml_name}/predictions/{yaml_name}"

    boltz_results = []
    for i in range(expected_result_count):
        with open(os.path.join(preds_dir, f"{yaml_name}_model_{i}.pdb"), 'r') as pdbfile:
            pdb_text = pdbfile.read()
            
        with open(os.path.join(preds_dir, f"confidence_{yaml_name}_model_{i}.json"),'r') as cf:
            confidence = json.load(cf)
        
        plddt = np.load(
            os.path.join(preds_dir, f"plddt_{yaml_name}_model_{i}.npz")
        )['plddt']
        
        this_out_dict = {
            'pdb':pdb_text,
            'confidence':confidence,
            'plddt':plddt
        }
        boltz_results.append(this_out_dict)

    return boltz_results

@mlflow.trace(span_type='TOOL')
def process_boltz_inputs(
    config: Dict,
    boltz_yaml_file_path: str,
    tmp_file_path: str,
    sequences: Dict[str, List[Tuple]],
    # protein_sequences: Optional[List[str]] = None,
    msa_file_paths: Optional[List[str]] = None,
    # dna_sequences: Optional[List[str]] = None,
    # rna_sequences: Optional[List[str]] = None,
    # small_molecule_sequences: Optional[List[str]] = None,
    cache: Optional[str] = None
    ):
    # if small_molecule_sequences is not None or nucleotide_sequences is not None or :
    #     raise NotImplementedError("nucleotide_sequences and small_molecule_sequences are not supported yet.")

    protein_sequences = sequences.get('protein', None)

    if msa_file_paths is not None and protein_sequences is not None:
        if len(msa_file_paths) != len(protein_sequences):
            raise ValueError("The number of msa_file_paths must be the same as the number of protien_sequences")

    def process_single_protein_chain(
        chain_id, 
        seq, 
        msa_file_path,
        ):
        out_d = {"protein":
                {
                    "id": chain_id,
                    "sequence": seq,
                }
                }
        if msa_file_path is not None:
            out_d['protein'].update({"msa": msa_file_path})
        return out_d
    
    with mlflow.start_span("Boltz input dict", span_type='TOOL') as span:
        span.set_inputs({"sequences": sequences, "msa_paths": msa_file_paths})
        input_dict = {'sequences': []}

        if protein_sequences is not None:
            if msa_file_paths is not None:
                for i,(s,msa_fp) in enumerate(zip(protein_sequences, msa_file_paths)):
                    # set chain_id to be letter of alphabet at position i
                    input_dict["sequences"].append(
                        process_single_protein_chain(
                            f"{str(list(s[0]))}",
                            s[1], 
                            msa_fp
                            )
                    )
            else:
                for i,s in enumerate(protein_sequences):
                    input_dict["sequences"].append(
                        process_single_protein_chain(
                            f"{str(list(s[0]))}",
                            s[1],
                            None
                            )
                    )
        naming_conv = {
            'dna':'sequence',
            'rna':'sequence',
            'ligand':'smiles'
        }
        for t in ['dna', 'rna', 'ligand']:
            seqs = sequences.get(t, None)
            if seqs:
                for s in seqs:
                    input_dict["sequences"].append({
                        t: {
                            'id': f"{str(list(s[0]))}",
                            naming_conv[t]: s[1],
                        }
                    })
        span.set_outputs({"Boltz input yaml content": input_dict})

    with open(boltz_yaml_file_path, 'w') as file:
        yaml.dump(input_dict, file)

    arg = boltz_yaml_file_path
    kwargs = {
        'out_dir' : tmp_file_path,
        'devices' : 1,
        'accelerator' : config['compute_type'], 
        'recycling_steps' : config['recycling_steps'],
        'sampling_steps' : config['sampling_steps'],
        'diffusion_samples' : config['diffusion_samples'],
        'output_format' : "pdb"
    }
    if cache:
        kwargs.update({'cache': cache})

    in_list = [arg]
    for k, v in kwargs.items():
        in_list.append('--'+k)
        in_list.append(v)

    if msa_file_paths is None:
        in_list.append('--use_msa_server')
    
    return in_list


@mlflow.trace(span_type='CHAIN')
def run_boltz(
    sequences : Dict[str, List[Tuple]], 
    config: Dict,
    ):
    """ Run vectorboltz protein

    Args:
        sequences : Dict with optional keys: ['protein', 'ligand', 'dna', 'rna'], and values lists of tuples (ids, sequence), ids is a tuple of ids
        config : A dictionary of model configuration parameters.
    """
    
    with tempfile.NamedTemporaryFile(suffix='.yaml') as f, \
         tempfile.TemporaryDirectory() as tmp_outdir:

        # parse out any jackhmmer kwargs
        jh_kwargs = {k.split('jh__')[1]:v for k,v in config.items() if k.startswith('jh__')}

        if config['msa']=='jh':
            msas=[]
            for sequence in sequences['protein']:
                msa_text = get_jackhmmer_alignment(
                    query=sequence[1], 
                    sequences=config['index_name'], 
                    jackhmmer_binary_path=config['jackhmmer_binary_path'],
                    jh_kwargs=jh_kwargs,
                )
                msas.append(msa_text)

        elif config['msa']=='no_msa':
            msas = []
            for sequence in sequences['protein']:
                msa_text = "empty" #f">protein\n{sequence[1]}\n"
                msas.append(msa_text)
        elif config['msa']=='mmseqs':
            msas = []
        else:
            raise ValueError("msa must be one of ['jh', 'no_msa', 'mmseqs']")

        # write msas to file for each sequence
        # I need to place in a named temp dir instead
        with tempfile.TemporaryDirectory() as tmp_dir:
            msa_paths = []
            for i, (s,msa_text) in enumerate(zip(sequences['protein'], msas)):
                if msa_text=='empty':
                    msa_paths.append(msa_text)
                else:
                    id_ = s[0] # this is a tuple of ids
                    chain = id_[0] # we only use the first - ok because there is only one a3m per sequence (not per id)
                    tmp_f = os.path.join(tmp_dir, f"{chain}.a3m")
                    msa_paths.append(tmp_f)
                    with open(tmp_f, 'w') as tmp_f_write:
                        tmp_f_write.write(msa_text)

            if len(msas)==0:
                msa_paths=None
                if config['msa']!='mmseqs':
                    raise ValueError(f"No msa sequences generated, this should not occur unless msa is set to 'mmseqs', it is set to {params['msa']}")

            in_list = process_boltz_inputs(
                config=config,
                boltz_yaml_file_path= f.name,
                tmp_file_path= tmp_outdir,
                sequences=sequences,
                msa_file_paths=msa_paths,
                cache=config.get('cache')
            )

            # use span so can log other variables too
            with mlflow.start_span("Boltz-1", span_type='LLM') as span:
                span.set_inputs({"input kwargs": in_list, "sequences": sequences})
                boltz_predict(in_list, standalone_mode=False)
                span.set_outputs({"Boltz-1 raw results": tmp_outdir})
            
            yaml_name = f.name.split(os.sep)[-1].split('.')[0]
            boltz_results = post_process_boltz_results(
                tmp_outdir, 
                yaml_name, 
                expected_result_count=config['diffusion_samples']
            )

        return boltz_results

def place_plddt_in_pdb(pdb : str, plddt : np.ndarray) -> str:
    pass

class Boltz(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.artifacts = context.artifacts
        self.model_config = context.model_config
        # NOTE: eventaully want to reqrite Boltz so that model load to GPU happens
        # only at this point and not on inference...
    
    def _prep_input_sequences(self, model_input : str) -> Dict[str, List[Tuple]]:
        def _get_ids(key):
            return key.split('_')[1].split(',')
        types = [
            'protein',
            'rna',
            'dna',
            'ligand'
        ]
        tmpdict = dict()
        for entry in model_input.split(';'):
            entry_ = entry.split(':')
            seq=entry_[1]
            key=entry_[0]
            tmpdict[key] = seq

        sequences = defaultdict(list)
        for k,v in tmpdict.items():
            for t in types:
                if k.startswith(t):
                    ids = _get_ids(k)
                    sequences[t].append( (tuple(ids), v) ) 
        return sequences
    
    def _prep_input_params(self,model_input):
        params = {k:v for k,v in model_input.items() 
                if (not k.startswith('protein')) and not (k.startswith('rna')) and not (k.startswith('dna')) and not (k.startswith('ligand'))
        }
        for k,v in params.items():
            if k in INT_INPUTS:
                params[k] = int(v)
            elif k in FLOAT_INPUTS:
                params[k] = float(v)
            elif k in BOOL_INPUTS:
                params[k] = v=='True'
            else:
                pass
        return params
    
    def _enforce_out_schema(self, results):
        new_results = []
        for r in results:
            # TODO: place plddt per residue in pdb - but have to be careful - per residue and also multi chain, rna, ligand etc
            # new_pdb = place_plddt_in_pdb(r['pdb'], r['plddt'])
            new_pdb = r['pdb']
            tmp_r = {}
            tmp_r['pdb'] = new_pdb
            for k,v in r['confidence'].items():
                if k in CONFIDENCE_ENTRIES_KEEP_SERVING:
                    tmp_r[k] = "{:.7f}".format(v)
            new_results.append(tmp_r)
        return new_results

    def predict(self, context, model_input: List[Dict[str,str]], params:Optional[Dict[str,Any]]=None) -> List[Dict[str,str]]:
        """ predicts one structure specification on each call - can be multipledissuion samoples out though

        Args:
            model_input: A list of protein/nucleotide/small_molecule sequences in one single structure. Each sequence can be formatted as "{entity_name}_{sequence}", e.g "protein_CASTTR", "dna_CCGGAT", "rna_UCG", "smiles_C1CCCCC1". If no entityis provided, assumes protein.
            
            params: dictionary of parameters for the model - can be chosen at runtime. includes: 'msa': 'vs' (default), 'jh' or 'no_msa', 'l2_distance_threshold': 2.0 (default), 'jh__evalue', 'jh__filter_f{1/2/3}".

        
        Returns:
            A list of dictionaries containing the structure and confidence scores, one list entry for each diffusion sample.

        """ 
        if len(model_input)>1:
            raise ValueError("Only one sequence at a time")

        model_input = model_input[0]
        
        params = {k:v for k,v in model_input.items() if k!='input'}
        model_input = model_input['input']
        
        # if params is None:
        #     params = DEFAULT_PARAMS.copy()

        sequences = self._prep_input_sequences(model_input)
            
        parsed_params = self._prep_input_params(params)
        # update params with user-provided params
        params_ = DEFAULT_PARAMS.copy()
        # use msa type provided, but default to none if not
        msa_type = parsed_params.get('msa', 'no_msa')
        # add default jh params if not given
        params_.update(DEFAULT_JH_PARAMS[msa_type])
        # now do update
        if params is not None:
            params_.update(parsed_params)
        
        # In future may want to not use predict() and instead ensure wights are alsways on GPU for serving...

        if 'cache' not in parsed_params:
            params_.update({'cache':self.artifacts['CACHE_DIR']})

        # allow user to overwrite config (copy) during inference (to decide how good an idea this is..)
        mc = self.model_config.copy()
        mc.update(params_)

        # run the model with desired params, inc. msa type
        boltz_results = run_boltz(
            sequences,
            config = mc,
        )
        boltz_results = self._enforce_out_schema(boltz_results)
        return boltz_results
                





















