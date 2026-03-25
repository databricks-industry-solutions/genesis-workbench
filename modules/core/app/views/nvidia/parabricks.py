import streamlit as st
import os
from pathlib import Path
import streamlit.components.v1 as components


st.title(":material/genetics: Parabricks")

@st.dialog("Sample Parabricks Commands", width="large")
def show_sample_notebook(): 
    with st.spinner("Loading.."):
        with open("views/nvidia/run_parabricks.html", 'r', encoding='utf-8') as file:
            file_content = file.read()
        
            if len(file_content) > 0:                            
                html_content=f"<iframe style='width: 100%; height: 600px border: None;' srcdoc='{file_content}'></iframe>"
                components.html(html_content, height=600)
            else:
                st.markdown("File not found")


st.divider()
st.markdown(f"A Parabricks enabled cluster is created for you")
st.markdown(f"Cluster Name: `{os.environ['PARABRICKS_CLUSTER_NAME']}` ")

st.markdown(f"You can run Pararicks commands in any notebook attached to the above cluster.")
st.markdown("""
Example:
```
data_vol = "/Volumes/my_catalog/my_schema/my_volume/data/"
```
```            
!pbrun germline \\
    --ref {data_vol}/parabricks_sample/Ref/Homo_sapiens_assembly38.fasta  \\
    --in-fq {data_vol}/parabricks_sample/Data/sample_1.fq.gz {data_vol}/parabricks_sample/Data/sample_2.fq.gz \\
    --knownSites {data_vol}/parabricks_sample/Ref/Homo_sapiens_assembly38.known_indels.vcf.gz \\
    --out-bam {data_vol}/output.bam \\
    --out-variants {data_vol}/germline.vcf \\
    --out-recal-file {data_vol}/recal.txt
```
""")

# view_parabricks_sample = st.button("See In action")
# if view_parabricks_sample:
#     show_sample_notebook()

# st.link_button("Open Notebook", url="/app/static/run_parabricks.html")

