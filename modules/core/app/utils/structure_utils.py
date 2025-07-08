import Bio.PDB as bp
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import numpy as np
import os
import tempfile

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO

class ChainSelect(bp.Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_residue(self, residue):
        """ remove hetatm """
        return 1 if residue.id[0] == " " else 0

    def accept_chain(self, chain):
        if chain.get_id() == self.chain:
            return 1
        else:          
            return 0
        
def _cif_to_pdb_str(cif_str:str)->str:
    with tempfile.TemporaryDirectory() as tmpdir:
        my_cif_path = os.path.join(tmpdir, 'my_cif.cif')
        my_pdb_path = os.path.join(tmpdir, 'my_pdb.cif')
        with open(my_cif_path, 'w') as f:
            f.write(cif_str)
        structure = MMCIFParser().get_structure('download',my_cif_path)
        io=PDBIO()
        io.set_structure(structure)
        io.save(my_pdb_path)
        with open(my_pdb_path,'r') as f:
            lines = f.readlines()
        pdb_str = ''.join(lines)
    return pdb_str

def get_seq_alignment(structure1, structure2):
    """ structures can be biopy.PDB.Structures, Models, or Chains """
    seq1 = ""
    seq2 = ""
    for residue in structure1.get_residues():
        seq1 += residue.get_resname()[0]
    for residue in structure2.get_residues():
        seq2 += residue.get_resname()[0]

    alignment = pairwise2.align.localxx(seq1, seq2)
    return alignment

def get_backbone_atoms(structure):
    backbone_atoms = []
    for atom in structure.get_atoms():
        if atom.get_name() in ['N', 'CA', 'C']:
            backbone_atoms.append(atom)
    return backbone_atoms

def get_overlapping_backbone(structure1, structure2):
    """ structures can be biopy.PDB.Structures, Models, or Chains """
    alignment = get_seq_alignment(structure1,structure2)

    seq=""
    a_count=0
    b_count=0
    a_residues=[r for r in structure1.get_residues()]
    b_residues=[r for r in structure2.get_residues()]
    a_backbone = []
    b_backbone = []

    for idx,(i,j) in enumerate(zip(alignment[0].seqA,alignment[0].seqB)):
        if i!='-' and j!='-':
            if i==j:
                seq+=i
            # print(a_residues[a_count].resname,b_residues[b_count].resname)
            a_backbone.extend(get_backbone_atoms(a_residues[a_count]))
            b_backbone.extend(get_backbone_atoms(b_residues[b_count]))
        if i!='-':
            a_count+=1
        if j!='-':
            b_count+=1
    return {'atoms1':a_backbone, 'atoms2':b_backbone, 'seq':seq} 

def pdb_to_str(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    return ''.join(lines)

def select_and_align(true_structure,af_structure):
    mmcif_parser = bp.MMCIFParser()
    pdb_parser = bp.PDBParser()

    alignment_scores = dict()
    for chain in true_structure.get_chains():
        ali = get_seq_alignment(af_structure, chain)[0]
        alignment_scores[chain.id] = ali.score
    max_overlapping_chain = max(alignment_scores, key=alignment_scores.get)

    # get Bio.PDB structure of just the chain of interest
    with tempfile.TemporaryDirectory() as tmp: 
        io = bp.PDBIO()
        io.set_structure(true_structure)
        new_name = os.path.join(tmp,"true_structure_singlechain.pdb")
        io.save(
            new_name,
            ChainSelect(max_overlapping_chain)
        )
        true_structure = pdb_parser.get_structure('true', new_name)

    alignment_dict = get_overlapping_backbone(true_structure, af_structure)
    ref_backbone_atoms = alignment_dict['atoms1']
    af_backbone_atoms = alignment_dict['atoms2']

    imposer = bp.Superimposer()
    imposer.set_atoms(
        ref_backbone_atoms,
        af_backbone_atoms 
    )
    imposer.apply(af_structure.get_atoms())

    with tempfile.TemporaryDirectory() as tmp:
        # create temp files of aligned structures
        io = bp.PDBIO()
        io.set_structure(af_structure)
        io.save(os.path.join(tmp,"af_s.pdb"))
        io.set_structure(true_structure)
        io.save(os.path.join(tmp,"tr_s.pdb"))
        af_structure_str = pdb_to_str(os.path.join(tmp,"af_s.pdb")) 
        true_structure_str = pdb_to_str(os.path.join(tmp,"tr_s.pdb")) 
    return true_structure_str, af_structure_str

    

