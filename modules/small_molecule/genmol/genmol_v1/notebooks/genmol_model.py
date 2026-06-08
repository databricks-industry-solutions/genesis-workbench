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
        import os, sys, shutil, tempfile, types

        # safe-mol's __init__ unconditionally `import wandb` (via .io). GenMol pins the
        # ancient wandb==0.13.5, which is incompatible with the serving env's modern stack
        # in MULTIPLE ways — sentry_sdk.Hub (removed in sentry 3.x), pkg_resources (removed
        # in setuptools 81+), and protobuf (wandb.proto.wandb_internal_pb2 has no attribute
        # 'Result' against mlflow's protobuf). Pinning around all of these is whack-a-mole
        # (and pinning protobuf down would break mlflow itself). wandb is only used for
        # training logging, never for inference — so install a minimal stub BEFORE importing
        # safe, so `import wandb` returns the stub and never executes wandb's real code.
        # Real dunders (__file__, etc.) are preserved so torch/inspect don't choke on it.
        if "wandb" not in sys.modules:
            import importlib.machinery
            _wandb = types.ModuleType("wandb")
            _wandb.__file__ = "wandb_stub.py"
            _wandb.__version__ = "0.13.5"
            # A real ModuleSpec so importlib.util.find_spec("wandb") works — accelerate
            # (imported by transformers.generation) calls it via is_wandb_available(),
            # and a None __spec__ raises ValueError. The real wandb dist-info is still on
            # disk (pip-installed), so importlib.metadata version checks also succeed.
            _wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
            def _wandb_getattr(name):
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)
                return lambda *a, **k: None
            _wandb.__getattr__ = _wandb_getattr
            sys.modules["wandb"] = _wandb

        # Model Serving containers have NO internet egress. GenMol's get_tokenizer()
        # pulls 'datamol-io/safe-gpt' from the HF hub at model init, so loading fails
        # in serving (it works on clusters, which have egress — that's why probes
        # passed). Force HF offline and monkeypatch get_tokenizer to load the bundled
        # tokenizer from the model artifact instead of the hub.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from safe.tokenizer import SAFETokenizer
        tok_dir = context.artifacts["safe_tokenizer"]

        def _local_get_tokenizer():
            tk = SAFETokenizer.from_pretrained(tok_dir).get_pretrained()
            tk.add_tokens(['<', '>'])  # mirror genmol.utils.utils_data.get_tokenizer
            return tk

        import genmol.utils.utils_data as _ud
        import genmol.model as _gm
        import genmol.sampler as gs
        for _m in (_ud, _gm, gs):
            if hasattr(_m, "get_tokenizer"):
                _m.get_tokenizer = _local_get_tokenizer

        # GenMol reads data/len.pk from genmol.sampler.ROOT_DIR (= dirname×3(sampler.py)
        # = site-packages, READ-ONLY in serving). Stage len.pk in a writable temp dir
        # and repoint the module-level ROOT_DIR — de_novo_generation reads
        # ROOT_DIR/data/len.pk via the module global, so reassigning redirects it.
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
