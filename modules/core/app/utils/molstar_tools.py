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
    return f"""<iframe style="width: 100%; height: 500px" border: None;" srcdoc='{html}'></iframe>"""

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
                </head>
                <body>
                    <div id="protein-viewer" style="width: 1200px; height: 400px; position: center"></div>
                    <script>
                        console.log("Initializing viewer...");
                        (async function() {{
                            // Create plugin instance
                            const viewer = new rcsbMolstar.Viewer("protein-viewer");

                            // PDB data in base64
                            const pdbData = "{pdb_base64}";

                            // Convert base64 to blob
                            const blob = new Blob(
                                [atob(pdbData)],
                                {{ type: "text/plain" }}
                            );

                            // Create object URL
                            const url = URL.createObjectURL(blob);

                            try {{
                                // Load structure
                                const structure = await viewer.loadStructureFromUrl(url, "pdb");
                                console.log("Structure loaded successfully");
                                //const model = structure.model;
                                //model.label = "some name";
                                //plugin.managers.structure.hierarchy.update(structure);
                                // Request a UI update
                                //plugin.managers.structure.updateStructure(structure);

                                // viewer.requestReprRender();
                            }} catch (error) {{
                                console.error("Error loading structure:", error);
                            }}
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

    html_str = f"""
            <!DOCTYPE html>
            <html>
                <head>
                    <script src="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.js"></script>
                    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.css">
                </head>
                <body>
                    <div id="protein-viewer" style="width: 1200px; height: 400px; position: center"></div>
                    <script>
                        console.log("Initializing viewer...");
                        (async function() {{
                            // Create plugin instance
                            const viewer = new rcsbMolstar.Viewer("protein-viewer");"""
    for i, pdb in enumerate(pdbs):
        pdb_base64 = base64.b64encode(pdb.encode()).decode()
        html_str += f"""
                            const pdbData_{i} = "{pdb_base64}";
                            const blob_{i} = new Blob(
                                                [atob(pdbData_{i})],
                                                {{ type: "text/plain" }}
                                            );
                            const url_{i} = URL.createObjectURL(blob_{i});"""

    html_str += """
                        try {"""
    for i, pdb in enumerate(pdbs):
        html_str += f"""
                            await viewer.loadStructureFromUrl(url_{i}, "pdb" );"""
    html_str += """
                        } catch (error) {
                            console.error("Error loading structure:", error);
                        }"""
        
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
    




