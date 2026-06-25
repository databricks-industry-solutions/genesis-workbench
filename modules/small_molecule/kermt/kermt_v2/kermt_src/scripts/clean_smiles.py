"""
Cleans a CSV file by removing rows with invalid SMILES strings and
canonicalizes valid SMILES strings.
"""

import os
import csv
import sys
from argparse import ArgumentParser
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog("rdApp.*")


def clean_csv(input_path, output_path, invalid_log_path=None, smiles_column=0):
    """
    Clean CSV file by removing invalid SMILES and canonicalizing valid ones.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output cleaned CSV file
        invalid_log_path: Optional path to save invalid SMILES log
        smiles_column: Column index containing SMILES (default: 0)
    """
    valid_count = 0
    invalid_count = 0
    invalid_smiles_list = []

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print("=" * 70)

    with open(input_path, "r") as f_in, open(output_path, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Copy header
        header = next(reader)
        writer.writerow(header)
        print(f"Header: {header}")
        print(f"SMILES column index: {smiles_column}")
        print("=" * 70)

        # Process each row
        for line_num, row in enumerate(
            reader, start=2
        ):  # Start at 2 (header is line 1)
            try:
                smiles = row[smiles_column]
                mol = Chem.MolFromSmiles(smiles)

                if mol is None:
                    invalid_count += 1
                    invalid_smiles_list.append((line_num, smiles))
                    print(f"Line {line_num}: INVALID SMILES - '{smiles}'")
                else:
                    # Canonicalize the SMILES
                    canonical_smiles = Chem.MolToSmiles(mol)
                    
                    # Replace original SMILES with canonical version
                    row[smiles_column] = canonical_smiles
                    
                    valid_count += 1
                    writer.writerow(row)

                    # Progress update every 100k rows
                    if valid_count % 100000 == 0:
                        print(
                            f"Processed {valid_count + invalid_count} rows... "
                            f"(Valid: {valid_count}, Invalid: {invalid_count})"
                        )

            except IndexError:
                print(
                    f"Line {line_num}: ERROR - Row has fewer than {smiles_column + 1} columns"
                )
                invalid_count += 1
            except Exception as e:
                print(f"Line {line_num}: ERROR - {e}")
                invalid_count += 1

    # Write invalid SMILES log if requested
    if invalid_log_path and invalid_smiles_list:
        with open(invalid_log_path, "w") as f_log:
            f_log.write("Line Number,SMILES\n")
            for line_num, smiles in invalid_smiles_list:
                f_log.write(f"{line_num},{smiles}\n")
        print(f"\nInvalid SMILES log saved to: {invalid_log_path}")

    # Print summary
    print("=" * 70)
    print("SUMMARY:")
    print(f"  Total rows processed: {valid_count + invalid_count}")
    print(
        f"  Valid SMILES:   {valid_count} ({100 * valid_count / (valid_count + invalid_count):.2f}%)"
    )
    print(
        f"  Invalid SMILES: {invalid_count} ({100 * invalid_count / (valid_count + invalid_count):.2f}%)"
    )
    print("=" * 70)
    print(f"✅ Cleaned and canonicalized CSV saved to: {output_path}")

    return valid_count, invalid_count


def main():
    parser = ArgumentParser(
        description="Clean CSV file by removing rows with invalid SMILES and canonicalizing valid SMILES",
        epilog="""
Examples:
  # Clean and canonicalize a CSV file (SMILES in first column)
  python clean_smiles.py --input data.csv --output data_clean.csv
  
  # Save invalid SMILES to a log file
  python clean_smiles.py --input data.csv --output data_clean.csv --invalid_log invalid.csv
  
  # SMILES in column 2 (index 1)
  python clean_smiles.py --input data.csv --output data_clean.csv --smiles_column 1
        """,
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output cleaned CSV file"
    )
    parser.add_argument(
        "--invalid_log",
        type=str,
        default=None,
        help="Path to save log of invalid SMILES (optional)",
    )
    parser.add_argument(
        "--smiles_column",
        type=int,
        default=0,
        help="Column index containing SMILES (default: 0)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)

    if os.path.exists(args.output):
        response = input(
            f"⚠️  Output file already exists: {args.output}\n   Overwrite? (y/n): "
        )
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Clean the CSV
    try:
        valid, invalid = clean_csv(
            args.input, args.output, args.invalid_log, args.smiles_column
        )

        if invalid == 0:
            print("\n🎉 Perfect! All SMILES were valid.")
        else:
            print(f"\n⚠️  Removed {invalid} invalid SMILES from the dataset.")

        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
