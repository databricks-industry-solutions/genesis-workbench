""" generate HTML for molstar viewer on the fly with pdb strings

html logic is adapted from:
https://github.com/modal-labs/modal-examples/tree/main

such that code in this py file are:
MIT License
Copyright (c) 2025 Databricks
Copyright (c) 2022 Modal Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import base64
from typing import Optional,List,Union
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOLSTAR_DARK_CSS = """
<style>
    body { background: #1e1e1e; margin: 0; }
    .msp-plugin { background: #1e1e1e !important; }
    .msp-plugin .msp-layout-static { background: #1e1e1e !important; }
    .msp-plugin .msp-scrollable-container { background: #252526 !important; }
    .msp-plugin .msp-btn { background: #333 !important; color: #ccc !important; }
    .msp-plugin .msp-form-control { background: #333 !important; color: #ccc !important; border-color: #555 !important; }
    .msp-plugin .msp-control-group-header { background: #2d2d2d !important; color: #ccc !important; }
    .msp-plugin .msp-section-header { color: #ccc !important; }
    .msp-plugin .msp-icon { color: #aaa !important; }
    .msp-plugin .msp-semi-transparent-background { background: rgba(30,30,30,0.8) !important; }
    .msp-plugin .msp-log-entry { color: #ccc !important; }
    .msp-plugin .msp-tree-row { color: #ccc !important; }
    .msp-plugin .msp-control-row { color: #ccc !important; }
    .msp-plugin .msp-accent-offset { background: #333 !important; }
    .msp-plugin .msp-representation-entry { background: #2d2d2d !important; }
</style>
"""

DUMMY_PDB = """HEADER    HYDROLASE                               01-JAN-00   1ABC
ATOM      1  N   ASP A  21      28.259  15.188   8.489  1.00 28.95           N
ATOM      2  CA  ASP A  21      28.698  14.971   7.110  1.00 27.84           C
ATOM      3  C   ASP A  21      29.944  15.799   6.813  1.00 26.75           C
ATOM      4  O   ASP A  21      30.013  16.995   7.108  1.00 26.40           O
ATOM      5  CB  ASP A  21      27.589  15.327   6.116  1.00 28.00           C
ATOM      6  CG  ASP A  21      26.301  14.582   6.390  1.00 29.27           C
ATOM      7  OD1 ASP A  21      26.163  13.971   7.469  1.00 30.12           O
ATOM      8  OD2 ASP A  21      25.410  14.654   5.515  1.00 30.95           O
END"""

def html_as_iframe(html : str) -> str:
    return f"""<iframe style="width: 100%; height: 520px; border: none;" srcdoc='{html}'></iframe>"""

def molstar_html_singlebody(pdb : str, name:Optional[str]=None, with_iframe:Optional[bool]=True) -> str:
    logging.info('doing single body molstar')
    pdb_base64 = base64.b64encode(pdb.encode()).decode()

    if name is None:
        label_dict={}
    else:
        label_dict = {"label": name}
        label_dict_str = str(label_dict).replace('\'label\'', 'label')

    # undefined, {label_dict_str} 
    html_str = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.js"></script>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.css">
            """ + MOLSTAR_DARK_CSS + f"""
        </head>
        <body>
            <div id="viewer" style="width: 100%; height: 500px;"></div>
            <script>
                (async function() {{
                    const viewer = new rcsbMolstar.Viewer("viewer", {{layoutShowLog: false, backgroundColor: 0x1e1e1e}});
                    await viewer.loadStructureFromData(atob("{pdb_base64}"), "pdb", false);
                }})();
            </script>
        </body>
    </html>"""
    if with_iframe:
        logging.info('wrap with iframe')
        return html_as_iframe(html_str)
    else:
        return html_str
    

def molstar_html_multibody(pdbs : Union[str, List[str]], names: Optional[Union[str, List[str]]]=None, with_iframe=True) -> str:
    if isinstance(pdbs, str):
        # pdbs = [pdbs]
        logging.info('sending from multi to single')
        return molstar_html_singlebody(pdbs, name=names, with_iframe=with_iframe)

    # list_label_dicts = []
    list_label_dicts_strs = []
    if names is None:
        list_label_dicts = ["{}"]*len(pdbs)
    else:
        if len(pdbs)!=len(names):
            logging.error('names must be same length as pdbs')
        for name in names:
            if name is not None:
                # list_label_dicts.append({"label": name})
                label_dict = {"label": name}
                list_label_dicts_strs.append(str(label_dict).replace('\'label\'', 'label'))
            else:
                list_label_dicts_strs.append("{}")
    # undefined,  {str(list_label_dicts[i])}

    html_str = """
    <!DOCTYPE html>
    <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.js"></script>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.css">
            """ + MOLSTAR_DARK_CSS + """
        </head>
        <body>
            <div id="viewer" style="width: 100%; height: 500px;"></div>
            <script>
                (async function() {
                    const viewer = new rcsbMolstar.Viewer("viewer", {layoutShowLog: false, backgroundColor: 0x1e1e1e});"""
    for i, pdb in enumerate(pdbs):
        pdb_base64 = base64.b64encode(pdb.encode()).decode()
        html_str += f"""
                    await viewer.loadStructureFromData(atob("{pdb_base64}"), "pdb", false);"""
    html_str += """
                })();
            </script>
        </body>
    </html>"""
    if with_iframe:
        logging.info('mulibody html going to iframe wrapping')
        logging.info(html_str.replace('\n', '____'))
        return html_as_iframe(html_str)
    else:
        return html_str


def molstar_html_complex(pdb: str, name: Optional[str] = None, with_iframe: Optional[bool] = True) -> str:
    """Render a multi-chain COMPLEX so it reads as a bound complex rather than two separate
    cartoons. Each chain gets a distinct color (Mol* `chain-id` theme) and a semi-transparent
    molecular surface is drawn over the cartoon, so a small interface — e.g. a nanobody's three
    CDR loops packed against an antigen epitope — visibly makes contact instead of looking like
    two detached blobs. `name` is accepted for signature parity with the single/multi helpers.

    Robustness: `loadStructureFromData` already applies a default (cartoon) preset, and the
    custom cartoon+surface build runs inside a try/catch — so if the representation API ever
    changes shape, the viewer degrades to the default preset instead of going blank. The JS uses
    double quotes only, so it survives both the React `srcDoc` and the single-quoted `srcdoc`
    iframe wrappers; the base64 is injected via replace() to avoid f-string brace escaping.
    (Representation recipe verified in-browser against an antibody-antigen complex.)"""
    logging.info('doing complex-aware molstar')
    pdb_base64 = base64.b64encode(pdb.encode()).decode()
    js = """
                (async function() {
                    const viewer = new rcsbMolstar.Viewer("viewer", {layoutShowLog: false, backgroundColor: 0x1e1e1e});
                    await viewer.loadStructureFromData(atob("__PDB_B64__"), "pdb", false);
                    // loadStructureFromData already draws a chain-id-colored cartoon. Overlay a
                    // semi-transparent molecular surface per chain so a small interface (e.g. a
                    // nanobody's three CDR loops against an antigen) reads as contact rather than two
                    // detached ribbons. Use the high-level structure-hierarchy manager (the raw
                    // state-tree walk fails because the Structure node is nested, not a root child),
                    // wrapped so any Mol* API drift degrades to the plain cartoon, never a blank viewer.
                    try {
                        await new Promise(function(r){ setTimeout(r, 400); });
                        const plugin = viewer.plugin;
                        const current = plugin.managers.structure.hierarchy.current;
                        for (const s of current.structures) {
                            for (const c of s.components) {
                                await plugin.builders.structure.representation.addRepresentation(
                                    c.cell, { type: "molecular-surface", typeParams: { alpha: 0.4 }, color: "chain-id" });
                            }
                        }
                    } catch (e) { console.log("surface overlay skipped, cartoon kept:", e && e.message); }
                })();
    """.replace("__PDB_B64__", pdb_base64)
    html_str = ("""
    <!DOCTYPE html>
    <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.js"></script>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.css">
            """ + MOLSTAR_DARK_CSS + """
        </head>
        <body>
            <div id="viewer" style="width: 100%; height: 500px;"></div>
            <script>""" + js + """</script>
        </body>
    </html>""")
    if with_iframe:
        return html_as_iframe(html_str)
    return html_str


def _pdb_to_mmcif_bfactor(pdb_str: str, res_values: Optional[dict] = None,
                          default_value: float = 0.0) -> str:
    """Minimal mmCIF (one `_atom_site` loop) from a PDB string, writing a per-residue value into
    `B_iso_or_equiv`. `res_values` maps (chain, resid) -> float; residues not listed get
    `default_value`. mmCIF is used deliberately (NOT PDB): rcsb-molstar only exposes B-factor to its
    color themes when the structure is loaded as mmCIF. All ATOM records are kept (so cartoon AND
    surface render); HETATM/waters are dropped."""
    rv = res_values or {}
    out = ["data_s", "#", "loop_",
           "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol", "_atom_site.label_atom_id",
           "_atom_site.label_comp_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id",
           "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z", "_atom_site.occupancy",
           "_atom_site.B_iso_or_equiv", "_atom_site.auth_seq_id", "_atom_site.auth_asym_id",
           "_atom_site.pdbx_PDB_model_num"]
    serial = 0
    for ln in pdb_str.splitlines():
        if ln[:4] != "ATOM" or len(ln) < 54:
            continue
        atom_name = ln[12:16].strip()
        comp = ln[17:20].strip() or "UNK"
        chain = ln[21]
        try:
            rid = int(ln[22:26]); x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
        except ValueError:
            continue
        occ = ln[54:60].strip() or "1.00"
        elem = (ln[76:78].strip() or atom_name[:1]) or "C"
        b = rv.get((chain, rid), default_value)
        serial += 1
        out.append(f"ATOM {serial} {elem} {atom_name} {comp} {chain} {rid} "
                   f"{x:.3f} {y:.3f} {z:.3f} {occ} {b:.2f} {rid} {chain} 1")
    out.append("#")
    return "\n".join(out) + "\n"


def molstar_html_colored(pdb_str: str, res_values: Optional[dict] = None, theme: str = "uncertainty",
                         with_surface: bool = False, with_iframe: bool = True) -> str:
    """Render a structure colored by a Mol* theme that reads a per-residue value from `B_iso_or_equiv`.

    Encodes `res_values` ((chain, resid) -> float) into the B-factor and loads as mmCIF (required so
    rcsb-molstar exposes the B-factor to the color themes — a plain PDB load does NOT). `theme`:
    - "plddt-confidence" — AlphaFold/ESMFold pLDDT palette (expects 0-100 in B-factor),
    - "uncertainty"      — continuous ramp; use it for coded categories (e.g. epitope/paratope,
                           CDR/framework) where distinct codes map to distinct ramp colors,
    - "chain-id"         — ignores B-factor, one color per chain (the complex "by chain" view).
    `with_surface` overlays a semi-transparent molecular surface (for complexes). The representation
    build is wrapped so any Mol* API drift degrades to the loaded structure, never a blank viewer.
    (Mechanism verified in-browser for pLDDT, epitope/paratope, and chain modes.)"""
    cif_b64 = base64.b64encode(_pdb_to_mmcif_bfactor(pdb_str, res_values).encode()).decode()
    surf_js = ("""
                        for (const s of cur.structures) { for (const c of s.components) {
                            await plugin.builders.structure.representation.addRepresentation(
                                c.cell, { type: "molecular-surface", typeParams: { alpha: 0.4 }, color: "__THEME__" });
                        } }""" if with_surface else "")
    js = ("""
                (async function() {
                    const viewer = new rcsbMolstar.Viewer("viewer", {layoutShowLog: false, backgroundColor: 0x1e1e1e});
                    await viewer.loadStructureFromData(atob("__CIF_B64__"), "mmcif", false);
                    try {
                        await new Promise(function(r){ setTimeout(r, 500); });
                        const plugin = viewer.plugin;
                        const cur = plugin.managers.structure.hierarchy.current;
                        for (const s of cur.structures) {
                            await plugin.managers.structure.component.updateRepresentationsTheme(
                                s.components, { color: "__THEME__" });
                        }__SURF__
                    } catch (e) { console.log("coloring skipped, default kept:", e && e.message); }
                })();
    """.replace("__CIF_B64__", cif_b64).replace("__SURF__", surf_js).replace("__THEME__", theme))
    html_str = ("""
    <!DOCTYPE html>
    <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.js"></script>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.css">
            """ + MOLSTAR_DARK_CSS + """
        </head>
        <body>
            <div id="viewer" style="width: 100%; height: 500px;"></div>
            <script>""" + js + """</script>
        </body>
    </html>""")
    if with_iframe:
        return html_as_iframe(html_str)
    return html_str

