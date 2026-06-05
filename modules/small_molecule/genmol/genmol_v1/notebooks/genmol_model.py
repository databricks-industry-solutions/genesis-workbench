"""GenMol MLflow PyFunc — logged via MLflow Models-from-Code.

Kept in its own module (not defined inline in the register notebook) because
mlflow.pyfunc cannot reliably cloudpickle a notebook __main__ class. log_model
receives this file path as `python_model`; `set_model(...)` at the bottom is the
Models-from-Code entry point. The PyFunc itself wraps genmol.sampler.Sampler.
"""
import mlflow
from mlflow.models import set_model


class GenMolGenerator(mlflow.pyfunc.PythonModel):
    """Wrap NVIDIA GenMol (Apache-2.0) for de novo / fragment-based generation.

    Input: single-column DataFrame of seed fragments. Empty string ⇒ de novo;
    a SMILES fragment ⇒ fragment_completion (scaffold decoration). GenMol returns
    canonical SMILES directly, so we RDKit-validate + score (QED/LogP).
    Output columns: seed, smiles, score.
    """

    def load_context(self, context):
        import os, shutil, tempfile
        import genmol.sampler as gs

        # GenMol reads data/len.pk from genmol.sampler.ROOT_DIR, which it computes
        # as dirname×3(sampler.py) = the package dir under site-packages. That dir
        # is READ-ONLY in Model Serving (writable on interactive clusters, which is
        # why probes passed). So stage len.pk in a writable temp dir and repoint the
        # module-level ROOT_DIR there — de_novo_generation reads ROOT_DIR/data/len.pk
        # via the module global, so reassigning it redirects the read.
        data_root = tempfile.mkdtemp(prefix="genmol_root_")
        os.makedirs(os.path.join(data_root, "data"), exist_ok=True)
        shutil.copy(context.artifacts["len_pk"], os.path.join(data_root, "data", "len.pk"))
        gs.ROOT_DIR = data_root

        # Sampler(path) loads the Lightning checkpoint; uses cuda if available.
        self.sampler = gs.Sampler(context.artifacts["checkpoint"])

    def _score(self, smiles, scoring):
        from rdkit import Chem
        from rdkit.Chem import QED, Crippen
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return float(Crippen.MolLogP(mol)) if scoring == "logp" else float(QED.qed(mol))

    def predict(self, context, model_input, params=None):
        import pandas as pd
        from rdkit import Chem

        params = params or {}
        n = int(params.get("num_molecules", 20))
        temperature = float(params.get("temperature", 1.0))
        randomness = float(params.get("randomness", 1.0))
        scoring = str(params.get("scoring", "qed")).lower()
        unique = bool(params.get("unique", True))

        if hasattr(model_input, "iloc"):
            seeds = model_input.iloc[:, 0].tolist()
        elif hasattr(model_input, "tolist"):
            seeds = model_input.tolist()
        else:
            seeds = list(model_input)
        if not seeds:
            seeds = [""]

        rows, seen = [], set()
        for seed in seeds:
            seed = (seed or "").strip()
            if seed:
                gens = self.sampler.fragment_completion(
                    seed, num_samples=n, softmax_temp=temperature, randomness=randomness)
            else:
                gens = self.sampler.de_novo_generation(
                    num_samples=n, softmax_temp=temperature, randomness=randomness)

            for smi in gens:
                if not isinstance(smi, str):
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                canonical = Chem.MolToSmiles(mol)
                if unique and canonical in seen:
                    continue
                seen.add(canonical)
                rows.append({
                    "seed": seed or "(de novo)",
                    "smiles": canonical,
                    "score": self._score(canonical, scoring),
                })

        result = pd.DataFrame(rows, columns=["seed", "smiles", "score"])
        if not result.empty:
            result = result.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
        return result


set_model(GenMolGenerator())
