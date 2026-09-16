"""Minimal pure-Python `torchtext` shim for scGPT.

torchtext is discontinued and its compiled `Vocab` (libtorchtext.so) is ABI-locked
to a specific torch build, so it cannot load against the serverless runtime's modern
torch (undefined-symbol OSError at model-load). scGPT only uses torchtext for a
token<->index gene vocabulary (`scgpt.tokenizer.gene_tokenizer.GeneVocab` subclasses
`torchtext.vocab.Vocab`), which needs no C++/torch at all. This shim reimplements that
tiny surface in pure Python so scGPT runs on any torch with no torchtext installed.

Injected via sys.path at registration and via mlflow code_paths at serving, so the
same package satisfies `import torchtext` / `import torchtext.vocab` in both places.
"""

__version__ = "0.18.0+scgpt-shim"

from . import vocab  # noqa: F401  (ensure `torchtext.vocab` is importable)
