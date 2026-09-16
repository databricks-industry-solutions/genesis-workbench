"""Pure-Python reimplementation of the `torchtext.vocab` surface scGPT uses.

scGPT's GeneVocab (scgpt/tokenizer/gene_tokenizer.py) does:
    import torchtext.vocab as torch_vocab
    from torchtext.vocab import Vocab
    class GeneVocab(Vocab): ...
        # __init__ builds `_vocab = torch_vocab.vocab(ordered_dict, ...)` then
        # calls super().__init__(_vocab.vocab)
        # from_dict(): cls([]) then insert_token(t, i) per entry, set_default_token
and callers use: __getitem__, __contains__, __len__, get_stoi, get_itos,
append_token, insert_token, set_default_index, lookup_indices/tokens.

Everything here is torch-free and ABI-free.
"""

from collections import Counter, OrderedDict

__all__ = ["Vocab", "vocab", "build_vocab_from_iterator"]


class _VocabData:
    """Stand-in for torchtext's C++ VocabPybind handle (`Vocab.vocab`)."""

    def __init__(self, tokens):
        self.tokens = list(tokens)


class Vocab:
    """Token<->index mapping matching the torchtext.vocab.Vocab API scGPT needs."""

    def __init__(self, vocab=None):
        # `vocab` may be: None (empty), a list/iterable of tokens, a dict
        # token->index, another Vocab, or a _VocabData (from `super().__init__(
        # _vocab.vocab)`).
        self._itos = []          # index -> token
        self._stoi = {}          # token -> index
        self._default_index = None
        if vocab is None:
            return
        if isinstance(vocab, Vocab):
            for t in vocab.get_itos():
                self._append(t)
            self._default_index = vocab.get_default_index()
        elif isinstance(vocab, _VocabData):
            for t in vocab.tokens:
                self._append(t)
        elif isinstance(vocab, dict):
            for t, _ in sorted(vocab.items(), key=lambda kv: kv[1]):
                self._append(t)
        else:  # assume iterable of tokens
            for t in vocab:
                self._append(t)

    # ---- internal ----------------------------------------------------------
    def _append(self, token):
        if token not in self._stoi:
            self._stoi[token] = len(self._itos)
            self._itos.append(token)

    def _reindex(self):
        self._stoi = {t: i for i, t in enumerate(self._itos)}

    # ---- torchtext.vocab.Vocab API ----------------------------------------
    def __len__(self):
        return len(self._itos)

    def __contains__(self, token):
        return token in self._stoi

    def __getitem__(self, token):
        idx = self._stoi.get(token, self._default_index)
        if idx is None:
            raise KeyError(token)
        return idx

    def get_stoi(self):
        return dict(self._stoi)

    def get_itos(self):
        return list(self._itos)

    def get_default_index(self):
        return self._default_index

    def set_default_index(self, index):
        self._default_index = index

    def append_token(self, token):
        if token in self._stoi:
            raise RuntimeError(f"Token {token!r} already exists in the Vocab")
        self._append(token)

    def insert_token(self, token, index):
        if token in self._stoi:
            raise RuntimeError(f"Token {token!r} already exists in the Vocab")
        self._itos.insert(index, token)
        self._reindex()

    def lookup_index(self, token):
        return self[token]

    def lookup_indices(self, tokens):
        return [self[t] for t in tokens]

    def lookup_token(self, index):
        return self._itos[index]

    def lookup_tokens(self, indices):
        return [self._itos[i] for i in indices]

    def forward(self, tokens):
        return self.lookup_indices(tokens)

    @property
    def vocab(self):
        # GeneVocab.__init__ passes `_vocab.vocab` to super().__init__.
        return _VocabData(list(self._itos))


def vocab(ordered_dict, min_freq=1, specials=None, special_first=True):
    """Factory mirroring torchtext.vocab.vocab(ordered_dict, ...)."""
    v = Vocab()
    tokens = [t for t, freq in ordered_dict.items() if freq >= min_freq]
    specials = list(specials) if specials else []
    if special_first:
        ordered = specials + [t for t in tokens if t not in specials]
    else:
        ordered = [t for t in tokens if t not in specials] + specials
    for t in ordered:
        v._append(t)
    return v


def build_vocab_from_iterator(iterator, min_freq=1, specials=None,
                              special_first=True, max_tokens=None):
    """Factory mirroring torchtext.vocab.build_vocab_from_iterator(...)."""
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    ordered = OrderedDict(
        sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    if max_tokens is not None:
        keep = max_tokens - (len(specials) if specials else 0)
        ordered = OrderedDict(list(ordered.items())[:keep])
    return vocab(ordered, min_freq=min_freq, specials=specials,
                 special_first=special_first)
