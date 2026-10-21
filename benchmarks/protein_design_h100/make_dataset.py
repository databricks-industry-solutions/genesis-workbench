# Databricks notebook source
"""Deterministic shared input set for the 3-way motif-scaffolding benchmark.

The benchmark's premise is "same data, everything the same." The three models don't share
a de-novo task (rfdiffusion = inpainting-only, proteina_complexa = target-conditional, only
rfd4 does de-novo), but they DO share **motif-scaffolding**: given a template backbone, keep
part of it fixed (the motif) and regenerate the rest.

Common spec per datapoint (model-agnostic):
    {sample_id, total_len N, motif_flank F, regen_start, regen_end, template_pdb}
keep flanks [1..F] and [N-F+1..N] as the motif, regenerate the contiguous middle
[F+1 .. N-F]. Each model notebook adapts this one spec into its own input:
  * rfdiffusion -> {pdb, start_idx=regen_start, end_idx=regen_end}
  * rfd4        -> task="motif", flanks fixed via condition_spec, generate the gap
  * proteina    -> AME motif-scaffolding on the same flanks

Templates are **deterministic idealized alpha-helix poly-glycine backbones** built here (no
network, reproducible, clash-free). For a speed/memory benchmark the coordinates' realism is
irrelevant — identical inputs + a clean length sweep (N drives compute) are what matter.

1000 datapoints by default, swept across a fixed length ladder so every model/run sees a
byte-identical sequence of (sample_id, N) and the plots can show time/memory-vs-length.
"""

from __future__ import annotations

import numpy as np

DEFAULT_LENGTHS: tuple[int, ...] = (60, 80, 100, 125, 150, 175, 200, 250, 300, 400)
MOTIF_FLANK: int = 15  # residues kept fixed at EACH terminus (30-residue motif total)

# Ideal backbone internal coordinates (standard values).
_BL = {"N_CA": 1.458, "CA_C": 1.525, "C_N": 1.329, "C_O": 1.231}
_BA = {"N_CA_C": 111.0, "CA_C_N": 116.0, "C_N_CA": 122.0, "CA_C_O": 120.5}  # degrees
_PHI, _PSI, _OMEGA = -57.0, -47.0, 180.0  # alpha-helix


def _nerf(a: np.ndarray, b: np.ndarray, c: np.ndarray, bond: float, angle_deg: float,
          dih_deg: float) -> np.ndarray:
    """Place atom d from the preceding three (a,b,c) via Natural Extension Reference Frame."""
    theta, chi = np.radians(angle_deg), np.radians(dih_deg)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.stack([bc, np.cross(n, bc), n], axis=1)
    d = np.array([-bond * np.cos(theta),
                  bond * np.sin(theta) * np.cos(chi),
                  bond * np.sin(theta) * np.sin(chi)])
    return c + m.dot(d)


def ideal_backbone(n_res: int) -> np.ndarray:
    """Return an (n_res, 4, 3) array of ideal alpha-helix backbone coords (N, CA, C, O)."""
    # Seed the first three main-chain atoms (N0, CA0, C0).
    N = [np.array([0.0, 0.0, 0.0])]
    CA = [np.array([_BL["N_CA"], 0.0, 0.0])]
    c0 = _nerf(np.array([-1.0, 1.0, 0.0]), N[0], CA[0], _BL["CA_C"], _BA["N_CA_C"], 0.0)
    C = [c0]
    O: list[np.ndarray] = []
    for i in range(1, n_res):
        n_i = _nerf(N[i - 1], CA[i - 1], C[i - 1], _BL["C_N"], _BA["CA_C_N"], _PSI)
        ca_i = _nerf(CA[i - 1], C[i - 1], n_i, _BL["N_CA"], _BA["C_N_CA"], _OMEGA)
        c_i = _nerf(C[i - 1], n_i, ca_i, _BL["CA_C"], _BA["N_CA_C"], _PHI)
        N.append(n_i)
        CA.append(ca_i)
        C.append(c_i)
    # Carbonyl O for each residue, placed off its own C from the residue's (N, CA, C) frame
    # (never collinear -> well-defined for every residue, including the last). A fixed
    # dihedral is fine: O realism doesn't affect a speed/memory benchmark.
    for i in range(n_res):
        o_i = _nerf(N[i], CA[i], C[i], _BL["C_O"], _BA["CA_C_O"], 180.0)
        O.append(o_i)
    return np.stack([np.stack(N), np.stack(CA), np.stack(C), np.stack(O)], axis=1)


def backbone_pdb(n_res: int, chain: str = "A") -> str:
    """A valid poly-glycine backbone PDB string of n_res residues (ideal alpha-helix)."""
    coords = ideal_backbone(n_res)
    names = ("N", "CA", "C", "O")
    elems = ("N", "C", "C", "O")
    lines, serial = [], 1
    for res_i in range(n_res):
        for a in range(4):
            x, y, z = coords[res_i, a]
            # Column-exact PDB ATOM record: resName 18-20, chainID at col 22, resSeq 23-26.
            lines.append(
                f"ATOM  {serial:>5} {names[a]:<4} GLY {chain}{res_i + 1:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}          {elems[a]:>2}"
            )
            serial += 1
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines)


def make_motif_dataset(n: int = 1000, lengths: tuple[int, ...] = DEFAULT_LENGTHS,
                       motif_flank: int = MOTIF_FLANK) -> list[dict]:
    """Return n deterministic motif-scaffolding specs (templates cached per unique length)."""
    cache: dict[int, str] = {}
    rows = []
    for i in range(n):
        total = int(lengths[i % len(lengths)])
        if total not in cache:
            cache[total] = backbone_pdb(total)
        rows.append({
            "sample_id": i,
            "total_len": total,
            "motif_flank": motif_flank,
            "regen_start": motif_flank + 1,          # 1-indexed, inclusive
            "regen_end": total - motif_flank,        # 1-indexed, inclusive
            "regen_len": total - 2 * motif_flank,
            "template_pdb": cache[total],
        })
    return rows


if __name__ == "__main__":
    ds = make_motif_dataset()
    from collections import Counter
    c = Counter(r["total_len"] for r in ds)
    r0 = ds[0]
    print(f"{len(ds)} motif specs across total_len {sorted(c)}; per-length counts {dict(sorted(c.items()))}")
    print(f"sample0: total_len={r0['total_len']} regen=[{r0['regen_start']}..{r0['regen_end']}] "
          f"regen_len={r0['regen_len']} pdb_atoms={r0['template_pdb'].count('ATOM')}")
