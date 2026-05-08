# NetSolP-1.0 distilled weights

This directory ships the two files needed to run the NetSolP-1.0 distilled
solubility predictor as a CPU serving endpoint:

- `Solubility_ESM12_0_quantized.onnx` — quantized ONNX model, split 0 of the upstream 5-fold ESM-12 ensemble (~85 MB; under GitHub's 100 MB hard limit)
- `ESM12_alphabet.pkl` — ESM-12 tokenization alphabet (<1 KB)

License: BSD 3-Clause (see `LICENSE.NETSOLP`). Both files are extracted from
the upstream `netsolp-1.0.ALL.tar.gz` distribution at
https://services.healthtech.dtu.dk/services/NetSolP-1.0/.

## One-time population

The two binary files are not in this repo by default — populate them with
`extract_weights.sh` before the first commit on a new clone:

1. Download `netsolp-1.0.ALL.tar.gz` (5.63 GB) from the DTU page above. By
   downloading you accept the BSD-3-Clause terms.
2. Run the extraction helper, pointing at the tarball:

   ```bash
   bash extract_weights.sh /path/to/netsolp-1.0.ALL.tar.gz
   ```

   This pulls only `S_Distilled_quantized.onnx` and `ESM12_alphabet.pkl` into
   this directory and discards the rest of the 5.63 GB bundle.
3. `git add weights/Solubility_ESM12_0_quantized.onnx weights/ESM12_alphabet.pkl`
   and commit.

## Why the single ESM-12 split, not the distilled model?

The upstream's recommended `Solubility_ESM1b_distilled_quantized.onnx` is **652 MB**
— well over GitHub's 100 MB per-file hard limit. The ESM-1b ensemble's individual
splits are also 652 MB each. Only the smaller ESM-12 backbone produces files
under the limit (~85 MB per split). We ship a single split here for repo
cleanliness; if a customer later needs the full 5-fold ensemble accuracy, they
can extract all five files (425 MB total) and update `REQUIRED_FILES` +
the PyFunc `predict()` to average the five splits.

After the commit the registration notebook deploys end-to-end with no manual
upload, on any clone of the repo.
