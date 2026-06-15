# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Copyright (c) 2021 Tencent AI Lab.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
The vocabulary building scripts.
"""
import os
from kermt.data.torchvocab import MolVocab, SMILESVocab
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Build vocabularies for KERMT pretraining")
    
    # Common arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to the data file (CSV or plain text with SMILES)")
    parser.add_argument('--vocab_save_folder', type=str, required=True,
                        help="Path to the folder where the vocab files will be saved")
    parser.add_argument('--dataset_name', type=str, default=None,
                        help="Prefix for vocab file names. If None, files will be: "
                             "atom_vocab.json, bond_vocab.json, smiles_vocab.pkl "
                             "(SMILES vocab is always pickle; see --vocab_format).")
    parser.add_argument('--vocab_format', type=str, default='json', choices=['json', 'pkl'],
                        help="Output format for the atom and bond vocab files. "
                             "Default 'json' uses secure JSON serialization (no arbitrary "
                             "code execution on load). 'pkl' opts back into pickle for "
                             "callers that explicitly need to interop with older consumers. "
                             "The SMILES vocab is always pickle (compiled regex state is "
                             "not JSON-serializable).")
    parser.add_argument('--num_workers', type=int, default=100,
                        help="Number of workers for parallel processing")
    
    # Vocabulary type selection
    vocab_type_group = parser.add_mutually_exclusive_group()
    vocab_type_group.add_argument('--build_all', action='store_true',
                                   help="Build all vocabularies (atom, bond, and SMILES) - DEFAULT")
    vocab_type_group.add_argument('--build_atom_bond_only', action='store_true',
                                   help="Build only atom and bond vocabularies")
    vocab_type_group.add_argument('--build_smiles_only', action='store_true',
                                   help="Build only SMILES vocabulary")
    
    # Atom/Bond vocabulary parameters
    atom_bond_group = parser.add_argument_group('Atom/Bond Vocabulary Parameters')
    atom_bond_group.add_argument('--vocab_max_size', type=int, default=None,
                                 help="Maximum vocabulary size for atom/bond vocabs (None = unlimited)")
    atom_bond_group.add_argument('--vocab_min_freq', type=int, default=1,
                                 help="Minimum frequency for atom/bond vocab inclusion")
    
    # SMILES vocabulary parameters
    smiles_group = parser.add_argument_group('SMILES Vocabulary Parameters')
    smiles_group.add_argument('--smiles_vocab_max_size', type=int, default=None,
                              help="Maximum vocabulary size for SMILES vocab (None = unlimited)")
    smiles_group.add_argument('--smiles_vocab_min_freq', type=int, default=1,
                              help="Minimum frequency for SMILES vocab inclusion")
    smiles_group.add_argument('--smiles_regex_path', type=str, default=None,
                              help="Path to SMILES regex pattern file. "
                                   "If None, uses default in kermt/data/smiles_regex.txt")
    
    args = parser.parse_args()
    
    # Default behavior: build all vocabularies if no specific option is selected
    if not (args.build_atom_bond_only or args.build_smiles_only):
        args.build_all = True
    
    return args

def build(args):
    """Build vocabularies based on user selection."""
    
    # Determine which vocabularies to build
    build_atom_bond = args.build_all or args.build_atom_bond_only
    build_smiles = args.build_all or args.build_smiles_only
    
    print("=" * 70)
    print("KERMT Vocabulary Building")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"Save folder: {args.vocab_save_folder}")
    print(f"Dataset name: {args.dataset_name if args.dataset_name else 'None (default names)'}")
    print(f"Workers: {args.num_workers}")
    print("\nBuilding:")
    print(f"  - Atom/Bond vocabularies: {'Yes' if build_atom_bond else 'No'}")
    print(f"  - SMILES vocabulary: {'Yes' if build_smiles else 'No'}")
    print("=" * 70)
    print()
    
    # Build atom and bond vocabularies
    if build_atom_bond:
        print("=" * 70)
        print("Building Atom and Bond Vocabularies")
        print("=" * 70)
        print(f"Max size: {args.vocab_max_size if args.vocab_max_size else 'Unlimited'}")
        print(f"Min frequency: {args.vocab_min_freq}")
        print()
        
        for vocab_type in ['atom', 'bond']:
            vocab_file = f"{vocab_type}_vocab.{args.vocab_format}"
            if args.dataset_name is not None:
                vocab_file = args.dataset_name + '_' + vocab_file
            vocab_save_path = os.path.join(args.vocab_save_folder, vocab_file)

            os.makedirs(os.path.dirname(vocab_save_path), exist_ok=True)
            
            print(f"Building {vocab_type} vocabulary...")
            vocab = MolVocab(file_path=args.data_path,
                             max_size=args.vocab_max_size,
                             min_freq=args.vocab_min_freq,
                             num_workers=args.num_workers,
                             vocab_type=vocab_type)
            print(f"  ✓ {vocab_type.capitalize()} vocab size: {len(vocab)}")
            vocab.save_vocab(vocab_save_path)
            print(f"  ✓ Saved to: {vocab_save_path}")
            print()
    
    # Build SMILES vocabulary
    if build_smiles:
        print("=" * 70)
        print("Building SMILES Vocabulary for Decoder Training")
        print("=" * 70)
        print(f"Max size: {args.smiles_vocab_max_size if args.smiles_vocab_max_size else 'Unlimited'}")
        print(f"Min frequency: {args.smiles_vocab_min_freq}")
        print(f"Regex file: {args.smiles_regex_path if args.smiles_regex_path else 'Default (kermt/data/smiles_regex.txt)'}")
        print()
        
        vocab_file = "smiles_vocab.pkl"
        if args.dataset_name is not None:
            vocab_file = args.dataset_name + '_' + vocab_file
        vocab_save_path = os.path.join(args.vocab_save_folder, vocab_file)
        
        os.makedirs(os.path.dirname(vocab_save_path), exist_ok=True)
        
        smiles_vocab = SMILESVocab(
            file_path=args.data_path,
            regex_path=args.smiles_regex_path,
            max_size=args.smiles_vocab_max_size,
            min_freq=args.smiles_vocab_min_freq,
            num_workers=args.num_workers
        )
        
        print(f"  ✓ SMILES vocab size: {len(smiles_vocab)}")
        print(f"  ✓ Special tokens: {[smiles_vocab.itos[i] for i in range(5)]}")
        print(f"  ✓ Sample tokens: {[smiles_vocab.itos[i] for i in range(5, min(15, len(smiles_vocab)))]}")
        
        smiles_vocab.save_vocab(vocab_save_path)
        print(f"  ✓ Saved to: {vocab_save_path}")
        print("=" * 70)
    
    print("\n✓ Vocabulary building completed successfully!\n")


if __name__ == '__main__':
    args = parse_args()
    build(args)
