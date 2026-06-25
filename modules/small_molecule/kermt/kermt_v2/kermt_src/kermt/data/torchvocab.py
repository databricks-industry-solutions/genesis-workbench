"""
The contextual property.
"""
import json
import pickle
import re
import os
from collections import Counter
from multiprocessing import Pool

import tqdm
from rdkit import Chem

from kermt.data.task_labels import atom_to_vocab
from kermt.data.task_labels import bond_to_vocab


# Special tokens for SMILES vocabulary
# <pad>: padding token
# <start>: start of sequence  
# <end>: end of sequence
# <unk>: unknown token
# <mask>: masking token (for MLM-style tasks if needed)
SMILES_SPECIAL_TOKENS = ('<pad>', '<start>', '<end>', '<unk>', '<mask>')


class TorchVocab(object):
    """
    Defines the vocabulary for atoms/bonds in molecular.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=('<pad>', '<other>'), vocab_type='atom'):
        """

        :param counter:
        :param max_size:
        :param min_freq:
        :param specials:
        :param vocab_type: 'atom': atom vocab; 'bond': bond vocab; 'smiles': SMILES vocab.
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        if vocab_type in ('atom', 'bond', 'smiles'):
            self.vocab_type = vocab_type
        else:
            raise ValueError(f"Wrong input for vocab_type! Got '{vocab_type}', expected 'atom', 'bond', or 'smiles'.")
        self.itos = list(specials)

        max_size = None if max_size is None else max_size + len(self.itos)
        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.other_index = 1
        self.pad_index = 0

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        # if self.vectors != other.vectors:
        #    return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
                self.freqs[w] = 0
            self.freqs[w] += v.freqs[w]

    def mol_to_seq(self, mol, with_len=False):
        mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
        if self.vocab_type == 'atom':
            seq = [self.stoi.get(atom_to_vocab(mol, atom), self.other_index) for i, atom in enumerate(mol.GetAtoms())]
        else:
            seq = [self.stoi.get(bond_to_vocab(mol, bond), self.other_index) for i, bond in enumerate(mol.GetBonds())]
        return (seq, len(seq)) if with_len else seq

    def to_dict(self):
        """Serialize this vocab to a JSON-safe dict.

        Note: subclasses that add non-JSON-safe state (e.g., compiled regex in
        :class:`SMILESVocab`) should either override save/load to stay on the
        pickle path or extend this method to include their state explicitly.
        """
        return {
            'freqs': dict(self.freqs),
            'itos': self.itos,
            'stoi': self.stoi,
            'vocab_type': self.vocab_type,
            'other_index': self.other_index,
            'pad_index': self.pad_index,
        }

    @classmethod
    def from_dict(cls, d):
        """Reconstruct a vocab from the dict produced by :meth:`to_dict`."""
        vocab = object.__new__(cls)
        vocab.freqs = Counter(d['freqs'])
        vocab.itos = d['itos']
        vocab.stoi = d['stoi']
        vocab.vocab_type = d['vocab_type']
        vocab.other_index = d['other_index']
        vocab.pad_index = d['pad_index']
        return vocab

    @classmethod
    def load_vocab(cls, vocab_path: str) -> 'TorchVocab':
        """Load a vocab, auto-detecting format from the file extension.

        Default is JSON (secure, no arbitrary-code-execution risk). Pickle
        is opt-in only by an explicit ``.pkl`` / ``.pckl`` extension, kept
        so older on-disk vocab files can still be loaded if encountered.
        """
        if vocab_path.endswith(('.pkl', '.pckl')):
            with open(vocab_path, "rb") as f:
                return pickle.load(f)
        with open(vocab_path, "r") as f:
            return cls.from_dict(json.load(f))

    def save_vocab(self, vocab_path):
        """Save a vocab, format chosen from the file extension.

        Default is JSON; ``.pkl`` / ``.pckl`` extensions opt into pickle
        (kept only for callers that explicitly need to interop with older
        pickle-only consumers).
        """
        if vocab_path.endswith(('.pkl', '.pckl')):
            with open(vocab_path, "wb") as f:
                pickle.dump(self, f)
            return
        with open(vocab_path, "w") as f:
            json.dump(self.to_dict(), f)


class MolVocab(TorchVocab):
    def __init__(self, smiles, max_size=None, min_freq=1, vocab_type='atom'):
        if vocab_type in ('atom', 'bond'):
            self.vocab_type = vocab_type
        else:
            raise ValueError('Wrong input for vocab_type!')

        print("Building %s vocab from smiles: %d" % (self.vocab_type, len(smiles)))
        counter = Counter()

        for smi in tqdm.tqdm(smiles):
            mol = Chem.MolFromSmiles(smi)
            if self.vocab_type == 'atom':
                for _, atom in enumerate(mol.GetAtoms()):
                    v = atom_to_vocab(mol, atom)
                    counter[v] += 1
            else:
                for _, bond in enumerate(mol.GetBonds()):
                    v = bond_to_vocab(mol, bond)
                    counter[v] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq, vocab_type=vocab_type)

    def __init__(self, file_path, max_size=None, min_freq=1, num_workers=1, total_lines=None, vocab_type='atom'):
        if vocab_type in ('atom', 'bond'):
            self.vocab_type = vocab_type
        else:
            raise ValueError('Wrong input for vocab_type!')
        print("Building %s vocab from file: %s" % (self.vocab_type, file_path))

        from rdkit import RDLogger
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)

        if total_lines is None:
            def file_len(fname):
                f_len = 0
                with open(fname) as f:
                    for f_len, _ in enumerate(f):
                        pass
                return f_len + 1

            total_lines = file_len(file_path)

        counter = Counter()
        pbar = tqdm.tqdm(total=total_lines)
        pool = Pool(num_workers)
        res = []
        batch = 50000
        callback = lambda a: pbar.update(batch)
        for i in range(int(total_lines / batch + 1)):
            start = int(batch * i)
            end = min(total_lines, batch * (i + 1))
            # print("Start: %d, End: %d"%(start, end))
            res.append(pool.apply_async(MolVocab.read_smiles_from_file,
                                        args=(file_path, start, end, vocab_type,),
                                        callback=callback))
            # read_smiles_from_file(lock, file_path, start, end)
        pool.close()
        pool.join()
        for r in res:
            sub_counter = r.get()
            for k in sub_counter:
                if k not in counter:
                    counter[k] = 0
                counter[k] += sub_counter[k]
        # print(counter)
        super().__init__(counter, max_size=max_size, min_freq=min_freq, vocab_type=vocab_type)

    @staticmethod
    def read_smiles_from_file(file_path, start, end, vocab_type):
        # print("start")
        smiles = open(file_path, "r")
        smiles.readline()
        sub_counter = Counter()
        for i, smi in enumerate(smiles):
            if i < start:
                continue
            if i >= end:
                break
            mol = Chem.MolFromSmiles(smi)
            # Skip invalid molecules (RDKit returns None for invalid SMILES)
            if mol is None:
                continue
            if vocab_type == 'atom':
                for atom in mol.GetAtoms():
                    v = atom_to_vocab(mol, atom)
                    sub_counter[v] += 1
            else:
                for bond in mol.GetBonds():
                    v = bond_to_vocab(mol, bond)
                    sub_counter[v] += 1
        # print("end")
        return sub_counter

    @classmethod
    def load_vocab(cls, vocab_path: str) -> 'MolVocab':
        """Load a MolVocab. Defaults to JSON; ``.pkl`` / ``.pckl`` opts into pickle.

        See :meth:`TorchVocab.load_vocab` for the extension convention.
        """
        if vocab_path.endswith(('.pkl', '.pckl')):
            with open(vocab_path, "rb") as f:
                return pickle.load(f)
        with open(vocab_path, "r") as f:
            return cls.from_dict(json.load(f))


class SMILESVocab(TorchVocab):
    """
    Vocabulary for SMILES tokenization using regex pattern.
    """
    
    def __init__(self, file_path, regex_path=None, max_size=None, min_freq=1, 
                 num_workers=1, total_lines=None):
        """
        Build SMILES vocabulary from file using regex tokenization.
        
        Args:
            file_path: Path to file containing SMILES strings (one per line or CSV)
            regex_path: Path to regex pattern file. If None, uses default in kermt/data/smiles_regex.txt
            max_size: Maximum vocabulary size
            min_freq: Minimum frequency for token inclusion
            num_workers: Number of parallel workers
            total_lines: Total number of lines (for progress bar)
        """
        self.vocab_type = 'smiles'
        
        # Load regex pattern
        if regex_path is None:
            # Use default regex file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            regex_path = os.path.join(current_dir, 'smiles_regex.txt')
        
        if not os.path.exists(regex_path):
            raise FileNotFoundError(f"Regex pattern file not found: {regex_path}")
        
        with open(regex_path, 'r') as f:
            regex_pattern = f.read().strip()
        
        self.regex = re.compile(r"(" + regex_pattern + r"|.)")
        
        print(f"Building SMILES vocab from file: {file_path}")
        print(f"Using regex pattern from: {regex_path}")
        
        # Count lines if not provided
        if total_lines is None:
            def file_len(fname):
                f_len = 0
                with open(fname) as f:
                    for f_len, _ in enumerate(f):
                        pass
                return f_len + 1
            total_lines = file_len(file_path)
        
        # Build counter using multiprocessing
        counter = Counter()
        pbar = tqdm.tqdm(total=total_lines)
        pool = Pool(num_workers)
        res = []
        batch = 50000
        callback = lambda a: pbar.update(batch)
        
        for i in range(int(total_lines / batch + 1)):
            start = int(batch * i)
            end = min(total_lines, batch * (i + 1))
            res.append(pool.apply_async(
                SMILESVocab.read_smiles_and_tokenize,
                args=(file_path, start, end, regex_pattern),
                callback=callback
            ))
        
        pool.close()
        pool.join()
        
        # Merge counters from all processes
        for r in res:
            sub_counter = r.get()
            for k in sub_counter:
                if k not in counter:
                    counter[k] = 0
                counter[k] += sub_counter[k]
        
        # Use module-level special tokens constant
        specials = SMILES_SPECIAL_TOKENS
        
        # Initialize parent class
        super().__init__(counter, max_size=max_size, min_freq=min_freq, 
                        specials=specials, vocab_type='smiles')
        
        # Update special token indices
        self.pad_index = self.stoi['<pad>']
        self.start_index = self.stoi['<start>']
        self.end_index = self.stoi['<end>']
        self.unk_index = self.stoi['<unk>']
        self.mask_index = self.stoi['<mask>']
        
        # Store regex for later use
        self.regex_pattern = regex_pattern
    
    @staticmethod
    def read_smiles_and_tokenize(file_path, start, end, regex_pattern):
        """
        Read SMILES from file and tokenize using regex.
        
        Args:
            file_path: Path to SMILES file
            start: Start line index
            end: End line index
            regex_pattern: Regex pattern string
        
        Returns:
            Counter object with token frequencies
        """
        regex = re.compile(r"(" + regex_pattern + r"|.)")
        sub_counter = Counter()
        
        with open(file_path, 'r') as f:
            # Skip header line if CSV
            first_line = f.readline()
            
            for i, line in enumerate(f):
                if i < start:
                    continue
                if i >= end:
                    break
                
                # Extract SMILES (handle both plain text and CSV)
                line = line.strip()
                if not line:
                    continue
                
                # If CSV, assume SMILES is first column (or handle comma-separated)
                if ',' in line:
                    smiles = line.split(',')[0].strip()
                else:
                    smiles = line
                
                # Tokenize using regex
                tokens = regex.findall(smiles)
                sub_counter.update(tokens)
        
        return sub_counter
    
    def tokenize(self, smiles):
        """
        Tokenize a SMILES string using the regex pattern.
        
        Args:
            smiles: SMILES string
        
        Returns:
            List of tokens
        """
        return self.regex.findall(smiles)
    
    def smiles_to_ids(self, smiles, add_special_tokens=True):
        """
        Convert SMILES string to token IDs.
        
        Args:
            smiles: SMILES string
            add_special_tokens: Whether to add <start> and <end> tokens
        
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(smiles)
        
        if add_special_tokens:
            tokens = ['<start>'] + tokens + ['<end>']
        
        ids = [self.stoi.get(token, self.unk_index) for token in tokens]
        return ids
    
    def ids_to_smiles(self, ids, skip_special_tokens=True):
        """
        Convert token IDs back to SMILES string.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            SMILES string
        """
        tokens = [self.itos.get(idx, '<unk>') for idx in ids]
        
        if skip_special_tokens:
            special_tokens = {'<pad>', '<start>', '<end>', '<unk>', '<mask>'}
            tokens = [t for t in tokens if t not in special_tokens]
        
        return ''.join(tokens)
    
    @staticmethod
    def load_vocab(vocab_path: str) -> 'SMILESVocab':
        """Load SMILES vocabulary from a pickle file.

        SMILESVocab holds a compiled regex pattern that cannot be JSON-
        serialized cleanly, so this class stays on the pickle path
        regardless of file extension. Atom and bond MolVocab files default
        to JSON; SMILESVocab files must be ``.pkl`` / ``.pckl``.
        """
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        """Save SMILES vocabulary to a pickle file.

        Overrides the TorchVocab extension-based dispatch because the
        compiled regex state is not JSON-serializable. Always pickles.
        """
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)
