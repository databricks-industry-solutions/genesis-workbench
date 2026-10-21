"""CPU-side structural decision metrics for the docking / ligand-binder viewers — no serving
endpoints, no heavy deps (pure-Python geometry; RDKit used lazily for ligand druglikeness).

Provides, from a protein PDB + a ligand HETATM/PDB block:
  * binding_site_pocket()  -> a combined 'pocket' PDB (binding-site residues + ligand) for the Mol*
                              interactions view, plus contact count + #site residues.
  * ligand_buried_sasa()   -> ligand solvent-accessible surface area buried on binding (A^2), via a
                              lightweight Shrake-Rupley on the (small) ligand heavy atoms.
  * mean_plddt_from_pdb()  -> mean CA B-factor (ESMFold writes pLDDT there).
  * ligand_druglikeness()  -> RDKit MW/logP/QED/TPSA/HBD/HBA/rot-bonds + Lipinski/Veber flags.
All functions degrade gracefully (return None / 0) rather than raising, so a viewer never breaks."""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Bondi van der Waals radii (A) for the common biomolecular elements; fallback 1.70 (carbon).
_VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
        "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "NA": 2.27, "MG": 1.73, "ZN": 1.39}
_PROBE = 1.4  # water probe radius (A)


def _elem(line: str, atom_name: str) -> str:
    e = line[76:78].strip().upper() if len(line) >= 78 else ""
    if not e:
        e = "".join(c for c in atom_name[:2] if c.isalpha()).upper()[:2] or "C"
    return e


def _parse(pdb: str, records):
    """Yield (record, chain, resid, resname, atom_name, x, y, z, element, raw_line)."""
    out = []
    for ln in pdb.splitlines():
        rec = ln[:6].strip()
        if rec not in records or len(ln) < 54:
            continue
        name = ln[12:16].strip()
        try:
            x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
        except ValueError:
            continue
        resid_s = ln[22:26].strip()
        try:
            resid = int(resid_s)
        except ValueError:
            resid = 0
        out.append((rec, ln[21], resid, ln[17:20].strip(), name, x, y, z, _elem(ln, name), ln))
    return out


def binding_site_pocket(protein_pdb: str, ligand_pdb: str, site_cutoff: float = 5.0,
                        contact_cutoff: float = 4.0):
    """Return (pocket_pdb, n_site_residues, contact_count). `pocket_pdb` = the verbatim ATOM lines of
    every protein residue with an atom within `site_cutoff` A of a ligand atom, plus the ligand block —
    loaded as one Mol* structure so an `interactions` representation finds ligand-residue H-bonds.
    `contact_count` = # heavy-atom protein-ligand pairs within `contact_cutoff` A."""
    prot = _parse(protein_pdb, ("ATOM",))
    lig = [a for a in _parse(ligand_pdb, ("HETATM", "ATOM")) if a[8] != "H"]
    if not prot or not lig:
        return "", 0, 0
    sc2, cc2 = site_cutoff * site_cutoff, contact_cutoff * contact_cutoff
    site_res = set()
    contacts = 0
    for (rec, ch, rid, rn, nm, x, y, z, el, raw) in prot:
        if el == "H":
            continue
        near_site = False
        for (_r, _c, _ri, _rn, _nm, lx, ly, lz, _le, _raw) in lig:
            d2 = (x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2
            if d2 <= cc2:
                contacts += 1
            if d2 <= sc2:
                near_site = True
        if near_site:
            site_res.add((ch, rid))
    pocket_lines = [a[9] for a in prot if (a[1], a[2]) in site_res]
    pocket_pdb = "\n".join(pocket_lines) + "\n" + ligand_pdb.rstrip() + "\nEND\n" if pocket_lines else ""
    return pocket_pdb, len(site_res), contacts


def _sphere_points(n: int):
    """n roughly-even points on the unit sphere (golden-spiral)."""
    pts = []
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - y * y))
        t = ga * i
        pts.append((math.cos(t) * r, y, math.sin(t) * r))
    return pts


def ligand_buried_sasa(protein_pdb: str, ligand_pdb: str, n_points: int = 92) -> Optional[float]:
    """Ligand SASA buried on binding (A^2) = SASA(ligand alone) - SASA(ligand in the protein context),
    via Shrake-Rupley on the ligand heavy atoms only (small count → cheap in pure Python). Protein
    occluders are limited to atoms within 10 A of the ligand. Returns None on failure."""
    try:
        lig = [a for a in _parse(ligand_pdb, ("HETATM", "ATOM")) if a[8] != "H"]
        if not lig:
            return None
        lig_xyz = [(a[5], a[6], a[7], _VDW.get(a[8], 1.70)) for a in lig]
        # Protein occluders = heavy atoms within 8 A of ANY ligand atom. Atoms farther than that cannot
        # block a water probe on a ligand atom, so this is both correct and much faster than a
        # whole-protein scan (keeps the SASA cost to ~ligand_atoms x points x a small occluder set).
        near2 = 8.0 * 8.0
        prot = []
        for a in _parse(protein_pdb, ("ATOM",)):
            if a[8] == "H":
                continue
            ax, ay, az = a[5], a[6], a[7]
            for (lx, ly, lz, _lr) in lig_xyz:
                if (ax - lx) ** 2 + (ay - ly) ** 2 + (az - lz) ** 2 <= near2:
                    prot.append((ax, ay, az, _VDW.get(a[8], 1.70)))
                    break
        sphere = _sphere_points(n_points)

        def sasa(occluders):
            total = 0.0
            for i, (ax, ay, az, ar) in enumerate(lig_xyz):
                rad = ar + _PROBE
                area_per = 4.0 * math.pi * rad * rad / n_points
                acc = 0
                for (sx, sy, sz) in sphere:
                    px, py, pz = ax + rad * sx, ay + rad * sy, az + rad * sz
                    buried = False
                    for j, (bx, by, bz, br) in enumerate(occluders):
                        if occluders is lig_xyz and j == i:
                            continue
                        rr = br + _PROBE
                        if (px - bx) ** 2 + (py - by) ** 2 + (pz - bz) ** 2 < rr * rr:
                            buried = True
                            break
                    if not buried:
                        acc += 1
                total += area_per * acc
            return total

        free = sasa(lig_xyz)
        complexed = sasa(lig_xyz + prot)
        return round(max(0.0, free - complexed), 1)
    except Exception as e:  # noqa: BLE001
        logger.info("ligand_buried_sasa failed: %s", e)
        return None


def mean_plddt_from_pdb(pdb: str) -> Optional[float]:
    """Mean CA B-factor (ESMFold writes per-residue pLDDT into the B-factor column)."""
    vals = []
    for ln in pdb.splitlines():
        if ln[:4] == "ATOM" and ln[12:16].strip() == "CA" and len(ln) >= 66:
            try:
                vals.append(float(ln[60:66]))
            except ValueError:
                continue
    return round(sum(vals) / len(vals), 1) if vals else None


def ligand_druglikeness(smiles: str) -> Optional[dict]:
    """RDKit physicochemical / druglikeness summary for a SMILES. None if RDKit unavailable or parse
    fails. Lipinski Ro5: <=1 violation of (MW<=500, logP<=5, HBD<=5, HBA<=10). Veber: rot<=10 & TPSA<=140."""
    if not smiles or not smiles.strip():
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    except Exception as e:  # noqa: BLE001
        logger.info("ligand_druglikeness: rdkit unavailable: %s", e)
        return None
    try:
        m = Chem.MolFromSmiles(smiles.strip())
        if m is None:
            return None
        mw = Descriptors.MolWt(m)
        logp = Crippen.MolLogP(m)
        hbd = Lipinski.NumHDonors(m)
        hba = Lipinski.NumHAcceptors(m)
        tpsa = rdMolDescriptors.CalcTPSA(m)
        rot = Lipinski.NumRotatableBonds(m)
        ro5_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        return {
            "mw": round(mw, 1), "logp": round(logp, 2), "qed": round(QED.qed(m), 3),
            "tpsa": round(tpsa, 1), "hbd": int(hbd), "hba": int(hba), "rotatable_bonds": int(rot),
            "lipinski_pass": bool(ro5_violations <= 1), "ro5_violations": int(ro5_violations),
            "veber_pass": bool(rot <= 10 and tpsa <= 140),
        }
    except Exception as e:  # noqa: BLE001
        logger.info("ligand_druglikeness failed: %s", e)
        return None
