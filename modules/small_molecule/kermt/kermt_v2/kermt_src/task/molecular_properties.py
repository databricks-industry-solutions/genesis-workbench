#!/usr/bin/env python
"""
Compute molecular properties from SMILES using RDKit.

This script computes various physicochemical, structural, and druglikeness
properties from SMILES strings. These properties can be used as labels for
probing classifiers to evaluate what information is encoded in latent representations.

================================================================================
Computed Properties:
================================================================================
Physicochemical:
    - MolecularWeight: Molecular weight
    - LogP: Octanol-water partition coefficient (Wildman-Crippen)
    - TPSA: Topological polar surface area
    - NumHDonors: Number of hydrogen bond donors
    - NumHAcceptors: Number of hydrogen bond acceptors
    - NumRotatableBonds: Number of rotatable bonds

Structural:
    - NumAtoms: Total number of atoms (including H)
    - NumHeavyAtoms: Number of heavy (non-hydrogen) atoms
    - NumRings: Number of rings
    - NumAromaticRings: Number of aromatic rings
    - NumHeteroatoms: Number of heteroatoms (non-C, non-H)
    - FractionCSP3: Fraction of sp3 carbons

Complexity:
    - NumBonds: Total number of bonds
    - NumStereocenters: Number of stereocenters
    - BertzCT: Bertz complexity index

Druglikeness (binary):
    - Lipinski: Passes Lipinski's Rule of 5
    - Veber: Passes Veber's rules (oral bioavailability)

================================================================================
Usage:
================================================================================
    # Single SMILES
    python task/molecular_properties.py -s "CCO"

    # Batch from CSV file
    python task/molecular_properties.py -i molecules.csv -o properties.csv

    # Select specific properties
    python task/molecular_properties.py -i molecules.csv -o properties.csv \
        --properties MolecularWeight LogP TPSA

    # Get property statistics
    python task/molecular_properties.py -i molecules.csv --stats
"""

import argparse
import csv
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import statistics

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.logger().setLevel(RDLogger.CRITICAL)


@dataclass
class PropertyResult:
    """Container for computed molecular properties."""
    smiles: str
    canonical_smiles: Optional[str]
    valid: bool
    properties: Dict[str, Any]


# Define all available properties and their computation functions
PROPERTY_FUNCTIONS = {
    # Physicochemical
    "MolecularWeight": lambda mol: Descriptors.MolWt(mol),
    "LogP": lambda mol: Descriptors.MolLogP(mol),
    "TPSA": lambda mol: Descriptors.TPSA(mol),
    "NumHDonors": lambda mol: Lipinski.NumHDonors(mol),
    "NumHAcceptors": lambda mol: Lipinski.NumHAcceptors(mol),
    "NumRotatableBonds": lambda mol: Lipinski.NumRotatableBonds(mol),
    
    # Structural
    "NumAtoms": lambda mol: mol.GetNumAtoms(),
    "NumHeavyAtoms": lambda mol: Lipinski.HeavyAtomCount(mol),
    "NumRings": lambda mol: Lipinski.RingCount(mol),
    "NumAromaticRings": lambda mol: rdMolDescriptors.CalcNumAromaticRings(mol),
    "NumHeteroatoms": lambda mol: Lipinski.NumHeteroatoms(mol),
    "FractionCSP3": lambda mol: Lipinski.FractionCSP3(mol),
    
    # Complexity
    "NumBonds": lambda mol: mol.GetNumBonds(),
    "NumStereocenters": lambda mol: len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    "BertzCT": lambda mol: Descriptors.BertzCT(mol),
    
    # Druglikeness (binary)
    "Lipinski": lambda mol: int(
        Descriptors.MolWt(mol) <= 500 and
        Descriptors.MolLogP(mol) <= 5 and
        Lipinski.NumHDonors(mol) <= 5 and
        Lipinski.NumHAcceptors(mol) <= 10
    ),
    "Veber": lambda mol: int(
        Descriptors.TPSA(mol) <= 140 and
        Lipinski.NumRotatableBonds(mol) <= 10
    ),
}

# Property categories for organization
PROPERTY_CATEGORIES = {
    "physicochemical": ["MolecularWeight", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"],
    "structural": ["NumAtoms", "NumHeavyAtoms", "NumRings", "NumAromaticRings", "NumHeteroatoms", "FractionCSP3"],
    "complexity": ["NumBonds", "NumStereocenters", "BertzCT"],
    "druglikeness": ["Lipinski", "Veber"],
}

# Properties suitable for classification (discrete values or can be binned)
CLASSIFICATION_PROPERTIES = [
    "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "NumRings", "NumAromaticRings", "NumHeteroatoms",
    "NumStereocenters", "Lipinski", "Veber"
]

# Properties suitable for regression (continuous values)
REGRESSION_PROPERTIES = [
    "MolecularWeight", "LogP", "TPSA", "FractionCSP3", "BertzCT"
]

ALL_PROPERTIES = list(PROPERTY_FUNCTIONS.keys())


def compute_properties(
    smiles: str,
    properties: Optional[List[str]] = None,
    add_hydrogens: bool = False
) -> PropertyResult:
    """
    Compute molecular properties for a SMILES string.

    Args:
        smiles: Input SMILES string
        properties: List of properties to compute (None = all)
        add_hydrogens: Whether to add explicit hydrogens before computing

    Returns:
        PropertyResult with computed properties
    """
    if properties is None:
        properties = ALL_PROPERTIES
    
    # Validate properties
    invalid_props = [p for p in properties if p not in PROPERTY_FUNCTIONS]
    if invalid_props:
        raise ValueError(f"Unknown properties: {invalid_props}. Available: {ALL_PROPERTIES}")
    
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return PropertyResult(
            smiles=smiles,
            canonical_smiles=None,
            valid=False,
            properties={p: None for p in properties}
        )
    
    # Canonicalize
    canonical = Chem.MolToSmiles(mol)
    
    # Optionally add hydrogens
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    
    # Compute properties
    computed = {}
    for prop in properties:
        try:
            computed[prop] = PROPERTY_FUNCTIONS[prop](mol)
        except Exception as e:
            print(f"Warning: Failed to compute {prop} for {smiles}: {e}", file=sys.stderr)
            computed[prop] = None
    
    return PropertyResult(
        smiles=smiles,
        canonical_smiles=canonical,
        valid=True,
        properties=computed
    )


def process_file(
    input_file: str,
    output_file: str,
    properties: Optional[List[str]] = None,
    smiles_column: int = 0,
    add_hydrogens: bool = False,
    include_invalid: bool = True
) -> Dict[str, Any]:
    """
    Process a CSV file and compute properties for all SMILES.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        properties: List of properties to compute (None = all)
        smiles_column: Column index containing SMILES (0-indexed)
        add_hydrogens: Whether to add explicit hydrogens
        include_invalid: Whether to include invalid SMILES in output

    Returns:
        Statistics dictionary
    """
    if properties is None:
        properties = ALL_PROPERTIES
    
    # Read input
    smiles_list = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row and len(row) > smiles_column:
                smiles_list.append(row[smiles_column].strip())
    
    print(f"Loaded {len(smiles_list)} SMILES from {input_file}")
    
    # Process
    results = []
    valid_count = 0
    invalid_count = 0
    
    for i, smiles in enumerate(smiles_list):
        if (i + 1) % 1000 == 0:
            print(f"Processing {i + 1}/{len(smiles_list)}...")
        
        result = compute_properties(smiles, properties, add_hydrogens)
        
        if result.valid:
            valid_count += 1
            results.append(result)
        else:
            invalid_count += 1
            if include_invalid:
                results.append(result)
    
    print(f"Valid: {valid_count}, Invalid: {invalid_count}")
    
    # Write output
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header_row = ["smiles", "canonical_smiles", "valid"] + properties
        writer.writerow(header_row)
        
        # Data
        for result in results:
            row = [result.smiles, result.canonical_smiles or "", result.valid]
            for prop in properties:
                val = result.properties.get(prop)
                row.append(val if val is not None else "")
            writer.writerow(row)
    
    print(f"Results written to {output_file}")
    
    # Compute statistics
    stats = compute_statistics(results, properties)
    
    return stats


def compute_statistics(results: List[PropertyResult], properties: List[str]) -> Dict[str, Any]:
    """Compute statistics for computed properties."""
    stats = {
        "total": len(results),
        "valid": sum(1 for r in results if r.valid),
        "invalid": sum(1 for r in results if not r.valid),
        "properties": {}
    }
    
    valid_results = [r for r in results if r.valid]
    
    for prop in properties:
        values = [r.properties[prop] for r in valid_results if r.properties[prop] is not None]
        
        if not values:
            stats["properties"][prop] = {"count": 0}
            continue
        
        # Check if numeric
        if all(isinstance(v, (int, float)) for v in values):
            prop_stats = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
            }
            if len(values) > 1:
                prop_stats["std"] = statistics.stdev(values)
            
            # For discrete properties, add value distribution
            if prop in CLASSIFICATION_PROPERTIES:
                value_counts = {}
                for v in values:
                    v_key = int(v) if isinstance(v, (int, float)) and v == int(v) else v
                    value_counts[v_key] = value_counts.get(v_key, 0) + 1
                # Sort by count and limit to top 10
                sorted_counts = sorted(value_counts.items(), key=lambda x: -x[1])[:10]
                prop_stats["value_distribution"] = dict(sorted_counts)
            
            stats["properties"][prop] = prop_stats
        else:
            stats["properties"][prop] = {"count": len(values), "type": "non-numeric"}
    
    return stats


def print_statistics(stats: Dict[str, Any]):
    """Print statistics in a formatted way."""
    print(f"\n{'=' * 70}")
    print("MOLECULAR PROPERTY STATISTICS")
    print(f"{'=' * 70}")
    
    print(f"\n--- Dataset Summary ---")
    print(f"  Total molecules:   {stats['total']}")
    print(f"  Valid molecules:   {stats['valid']} ({100 * stats['valid'] / stats['total']:.1f}%)")
    print(f"  Invalid molecules: {stats['invalid']} ({100 * stats['invalid'] / stats['total']:.1f}%)")
    
    for prop, prop_stats in stats["properties"].items():
        if prop_stats.get("count", 0) == 0:
            continue
        
        print(f"\n--- {prop} ---")
        print(f"  Count:  {prop_stats['count']}")
        
        if "mean" in prop_stats:
            print(f"  Min:    {prop_stats['min']:.4g}")
            print(f"  Max:    {prop_stats['max']:.4g}")
            print(f"  Mean:   {prop_stats['mean']:.4g}")
            print(f"  Median: {prop_stats['median']:.4g}")
            if "std" in prop_stats:
                print(f"  Std:    {prop_stats['std']:.4g}")
        
        if "value_distribution" in prop_stats:
            print(f"  Top values:")
            for val, count in list(prop_stats["value_distribution"].items())[:5]:
                pct = 100 * count / prop_stats["count"]
                print(f"    {val}: {count} ({pct:.1f}%)")
    
    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute molecular properties from SMILES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--smiles", "-s",
        type=str,
        help="Single SMILES string to analyze"
    )
    input_group.add_argument(
        "--input_file", "-i",
        type=str,
        help="Input CSV file with SMILES"
    )
    
    # Output
    parser.add_argument(
        "--output_file", "-o",
        type=str,
        default=None,
        help="Output CSV file (required for batch mode)"
    )
    
    # Property selection
    parser.add_argument(
        "--properties", "-p",
        nargs="+",
        default=None,
        help=f"Properties to compute. Available: {ALL_PROPERTIES}"
    )
    parser.add_argument(
        "--category", "-c",
        choices=list(PROPERTY_CATEGORIES.keys()),
        default=None,
        help="Compute all properties in a category"
    )
    
    # Options
    parser.add_argument(
        "--smiles_column",
        type=int,
        default=0,
        help="Column index containing SMILES (0-indexed)"
    )
    parser.add_argument(
        "--add_hydrogens",
        action="store_true",
        help="Add explicit hydrogens before computing properties"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics after processing"
    )
    parser.add_argument(
        "--list_properties",
        action="store_true",
        help="List all available properties and exit"
    )
    
    args = parser.parse_args()
    
    # List properties mode
    if args.list_properties:
        print("Available properties:")
        for category, props in PROPERTY_CATEGORIES.items():
            print(f"\n  {category.upper()}:")
            for prop in props:
                marker = "[C]" if prop in CLASSIFICATION_PROPERTIES else "[R]"
                print(f"    {marker} {prop}")
        print("\n  [C] = suitable for classification, [R] = suitable for regression")
        return
    
    # Determine properties to compute
    properties = args.properties
    if args.category:
        properties = PROPERTY_CATEGORIES[args.category]
    
    if args.smiles:
        # Single SMILES mode
        result = compute_properties(args.smiles, properties, args.add_hydrogens)
        
        print(f"\nSMILES: {result.smiles}")
        print(f"Canonical: {result.canonical_smiles}")
        print(f"Valid: {result.valid}")
        
        if result.valid:
            print(f"\nProperties:")
            for prop, val in result.properties.items():
                print(f"  {prop}: {val}")
    else:
        # Batch mode
        if not args.output_file:
            parser.error("--output_file is required with --input_file")
        
        stats = process_file(
            input_file=args.input_file,
            output_file=args.output_file,
            properties=properties,
            smiles_column=args.smiles_column,
            add_hydrogens=args.add_hydrogens
        )
        
        if args.stats:
            print_statistics(stats)


if __name__ == "__main__":
    main()
