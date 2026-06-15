"""
The data splitting script for pretraining.

Usage:
    # With features (original vocab-based pretraining):
    python split_data.py --data_path smiles.csv --features_path features.npz --output_path output/

    # Without features (CMIM pretraining):
    python split_data.py --data_path smiles.csv --output_path output/
"""
import os
from argparse import ArgumentParser
import csv
import shutil
import numpy as np


import kermt.util.utils as fea_utils


parser = ArgumentParser()
parser.add_argument("--data_path", required=True, help="Path to SMILES CSV file")
parser.add_argument("--features_path", default=None, help="Path to features .npz file (optional, skip for CMIM)")
parser.add_argument("--sample_per_file", type=int, default=1000)
parser.add_argument("--output_path", required=True, help="Output directory for split data")


def load_smiles(data_path):
    with open(data_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        res = []
        for line in reader:
            res.append(line)
    return res, header


def load_features(data_path):
    fea = fea_utils.load_features(data_path)
    return fea


def save_smiles(data_path, index, data, header):
    fn = os.path.join(data_path, str(index) + ".csv")
    with open(fn, "w") as f:
        fw = csv.writer(f)
        fw.writerow(header)
        for d in data:
            fw.writerow(d)


def save_features(data_path, index, data):
    fn = os.path.join(data_path, str(index) + ".npz")
    np.savez_compressed(fn, features=data)


def run():
    args = parser.parse_args()
    res, header = load_smiles(data_path=args.data_path)
    
    # Features are optional (not needed for CMIM pretraining)
    if args.features_path is not None:
        fea = load_features(data_path=args.features_path)
        assert len(res) == fea.shape[0], f"Feature count ({fea.shape[0]}) != SMILES count ({len(res)})"
        print("Features loaded: %d samples" % fea.shape[0])
    else:
        fea = None
        print("No features path provided - skipping feature processing (CMIM mode)")

    n_graphs = len(res)
    perm = np.random.permutation(n_graphs)

    nfold = (n_graphs + args.sample_per_file - 1) // args.sample_per_file
    print("Number of files: %d" % nfold)
    
    # Create output directory structure
    os.makedirs(args.output_path, exist_ok=True)
    graph_path = os.path.join(args.output_path, "graph")
    fea_path = os.path.join(args.output_path, "feature")
    
    # Only remove the specific subdirectories we'll recreate
    if os.path.exists(graph_path):
        shutil.rmtree(graph_path)
    if fea is not None and os.path.exists(fea_path):
        shutil.rmtree(fea_path)
    
    os.makedirs(graph_path, exist_ok=True)
    if fea is not None:
        os.makedirs(fea_path, exist_ok=True)

    for i in range(nfold):
        sidx = i * args.sample_per_file
        eidx = min((i + 1) * args.sample_per_file, n_graphs)
        indexes = perm[sidx:eidx]
        sres = [res[j] for j in indexes]
        save_smiles(graph_path, i, sres, header)
        if fea is not None:
            sfea = fea[indexes]
            save_features(fea_path, i, sfea)

    summary_path = os.path.join(args.output_path, "summary.txt")
    summary_fout = open(summary_path, 'w')
    summary_fout.write("n_files:%d\n" % nfold)
    summary_fout.write("n_samples:%d\n" % n_graphs)
    summary_fout.write("sample_per_file:%d\n" % args.sample_per_file)
    summary_fout.close()


if __name__ == "__main__":
    run()
