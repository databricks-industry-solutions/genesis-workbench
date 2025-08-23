import streamlit as st
import os
import json
from io import StringIO

from genesis_workbench.workbench import UserInfo
from genesis_workbench.models import (ModelCategory,                                       
                                      get_uc_model_info,
                                      import_model_from_uc,
                                      get_gwb_model_info,
                                      deploy_model)


from databricks.sdk import WorkspaceClient
from streamlit.components.v1 import html


def get_user_info():

    user_info = None

    if "__gwb_user_info" not in st.session_state:
        headers = st.context.headers
        user_access_token = headers.get("X-Forwarded-Access-Token")
        user_name=headers.get("X-Forwarded-Preferred-Username")
        user_display_name = ""
        user_groups = []

        user_groups_retrieved = False
        retry_count = 0
        if user_access_token:
            while retry_count <= 1 and not user_groups_retrieved:
                try:    
                    if user_access_token:
                        # Initialize WorkspaceClient with the user's token
                        w = WorkspaceClient(token=user_access_token, auth_type="pat")
                        # Get current user information
                        current_user = w.current_user.me()
                        # Display user information
                        user_display_name = current_user.display_name
                        user_groups = [g.display for g in current_user.groups]
                        print(f"User belongs to following groups: {user_groups}")
                        user_groups_retrieved = True

                except Exception as e:
                    print(f"Error getting user info: {e}")
                    retry_count += 1
                    if retry_count > 1:
                        raise e
                    else:
                        print(f"Retrying...")

        user_email = headers.get("X-Forwarded-Email")
        if not user_email or user_email.strip() == "":
            user_email = os.environ["USER_EMAIL"]

        user_info = UserInfo(
            user_name=user_name,
            user_email = user_email,
            user_id=headers.get("X-Forwarded-User"),
            user_access_token = headers.get("X-Forwarded-Access-Token"),
            user_display_name = user_display_name if user_display_name != "" else user_email,
            user_groups = user_groups
        )

        st.session_state["__gwb_user_info"] = user_info

    else:
        user_info = st.session_state["__gwb_user_info"]

    return user_info


def open_run_window(job_id,run_id):
    host_name = os.getenv("DATABRICKS_HOSTNAME")    
    url = f"{host_name}/jobs/{job_id}/runs/{run_id}"
    if not url.startswith("https://"):
        url = "https://" + url
        
    print(url)
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)

def open_mlflow_experiment_window(exeriment_id):
    host_name = os.getenv("DATABRICKS_HOSTNAME")    
    url = f"{host_name}/ml/experiments/{exeriment_id}/runs"
    if not url.startswith("https://"):
        url = "https://" + url
        
    print(url)
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)

def make_run_link(job_id, run_id):
    host_name = os.getenv("DATABRICKS_HOST")
    if not host_name.startswith("https://"):
        host_name = "https://" + host_name
    url = f"{host_name}/jobs/{job_id}/runs/{run_id}"
    return f"<a href='{url}' target='_blank' style='color: #1E90FF;'>{run_id}</a>"

@st.dialog("Deploy Model", width="large")
def display_deploy_model_dialog(selected_model_name, success_callback = None, error_callback = None):    
    """Dialog to deploy a model to model serving"""
    model_info = None
    run_id = None
    deploy_model_clicked = False
    view_deploy_run_btn = False
    close_deploy_run_btn = False
    user_info = get_user_info()
    is_adapter = False
    model_id = int(selected_model_name.split("-")[0].strip())

    with st.spinner("Getting model details"):
        try:
            model_info = get_gwb_model_info(model_id)        
            st.session_state["deployment_model_details"] = model_info
        except Exception as e:
            st.error("Error getting model details.")
            model_info = None
            if error_callback:
                error_callback()

    if model_info:
        model_details = model_info.model_uc_name.split(".") 
        st.write(f"Model Name: {model_info.model_display_name} ")
        st.write(f"Registered Name: {model_info.model_uc_name} v{model_info.model_uc_version}")
        
        if model_info.is_model_deployed:
            st.warning("❗️This model has existing deployment(s).")

        with st.form("deploy_model_details_form", enter_to_submit=False):
            deploy_name = st.text_input("Deployment Name:", placeholder="eg: finetuned geneformer")
            deploy_description= st.text_area("Deployment Description:", max_chars=5000)
            c1,c2,c3 = st.columns([1,1,1])
            with c1:
                st.write("Input Schema")
                st.json(model_info.model_input_schema, expanded=False)
            with c2:
                st.write("Output Schema")
                st.json(model_info.model_output_schema, expanded=False)
            with c3:
                st.write("Parameters")
                st.json(model_info.model_params_schema, expanded=False)
            
            with st.expander("Add Input/Output Adapters:"):
                c1,c2 = st.columns([1,1])
                with c1:
                    input_adapter_code_text = ""
                    input_adapter_code = st.file_uploader("Input Adapter:", type="py", help="A python file with only one class definition that extends `genesis_workbench.models.BaseAdapter`." )
                    if input_adapter_code is not None:
                        stringio = StringIO(input_adapter_code.getvalue().decode("utf-8"))
                        input_adapter_code_text = stringio.read()
                with c2: 
                    output_adapter_code_text = ""
                    output_adapter_code = st.file_uploader("Output Adapter:", type="py", help="A python file with only one class definition that extends `genesis_workbench.models.BaseAdapter`." )
                    if output_adapter_code is not None:
                        stringio = StringIO(output_adapter_code.getvalue().decode("utf-8"))
                        output_adapter_code_text = stringio.read()
                
                sample_input_data_dict_as_json = '{"data": [1.0, 2.0, 3.0, 4.0, 5.0],\
                    "type":"list" \
                    } '
                sample_params_as_json = '{"index": "a", "num_embeddings": 10}'
                new_sample_input = st.text_area("Provide a sample input data (required if using adapters):", help=f"Example: `{sample_input_data_dict_as_json}`")
                new_sample_params = st.text_area("Provide a sample parameter dictionary (required if using adapters):" ,help=f"Example: `{sample_params_as_json}`")

            compute_type = st.selectbox("Model Serving Compute:", ["CPU", "GPU_SMALL", "GPU_MEDIUM", "GPU_LARGE"])
            workload_size = st.selectbox("Workload Size:", ["Small", "Medium","Large"])            
            deploy_model_clicked = st.form_submit_button("Deploy Model")

        deploy_started = False
        validation_pass = True
        error_message = ""

        if deploy_model_clicked:

            with st.spinner("Launching deploy job"):
                #validate inputs
                if len(deploy_name.strip())>0 and len(deploy_description.strip())>0:

                    if (len(input_adapter_code_text.strip()) > 0 or 
                       len(output_adapter_code_text.strip()) > 0 or
                       len(new_sample_input.strip()) > 0 or 
                       len(new_sample_params.strip()) > 0) :

                        try:
                            json.loads(new_sample_input)
                        except Exception as e:
                            validation_pass = False
                            error_message = f"Unable to parse sample input JSON: \n {e}"

                        try:
                            json.loads(new_sample_params)
                        except Exception as e:
                            validation_pass = False
                            error_message = f"Unable to parse sample params JSON: \n {e}"

                    if validation_pass:
                        try:
                            run_id = deploy_model(user_email = user_info.user_email,
                                                gwb_model_id = model_id,
                                                deployment_name=deploy_name,
                                                deployment_description=deploy_description, 
                                                input_adapter_str=input_adapter_code_text,
                                                output_adapter_str=output_adapter_code_text,
                                                sample_input_data_dict_as_json=new_sample_input,
                                                sample_params_as_json=new_sample_params,
                                                workload_type=compute_type,
                                                workload_size=workload_size)

                            deploy_started = True
                            
                        except Exception as e:
                            print(e)
                            st.error("Error launching deploy job.")                
                            deploy_started = False
                            if error_callback:
                                error_callback()
                    else:
                        deploy_started = False
                    
                else:
                    validation_pass = False
                    error_message = f"Deployment name and description is required."


        if not validation_pass:
            st.error(error_message)
            if error_callback:
                error_callback()

        if deploy_started:
            st.success(f"Model deploy has started with run id: {run_id}.")                
            st.warning(f"It might take upto 30 minutes to complete")
            job_id = os.getenv("DEPLOY_MODEL_JOB_ID")
            view_deploy_run_btn = st.button("View Run", on_click=lambda: open_run_window(job_id,run_id))
            if success_callback:
                success_callback()



@st.dialog("Import model from Unity Catalog")
def display_import_model_uc_dialog(model_category: ModelCategory, success_callback = None, error_callback = None):    
    """Dialog to import a model from UC"""
    model_info = None
    model_info_error = False
    model_import_error = False
    fetch_model_info_clicked = False
    uc_import_model_clicked = False
    user_info = get_user_info()

    if "import_uc_model_info" in st.session_state:
        model_info = st.session_state["import_uc_model_info"]

    with st.form("import_model_uc_form_fetch", enter_to_submit=False ):
        c1,c2,c3 = st.columns([3,1,1], vertical_alignment="bottom")
        with c1:
            uc_model_name = st.text_input("Unity Catalog Name (catalog.schema.model_name):", value="genesis_workbench.dev_srijit_nair_dbx_genesis_workbench_core.gene_embedder")
        with c2:
            uc_model_version = st.number_input("Version:", min_value=1, step=1, max_value=999)
        with c3:
            fetch_model_info_clicked = st.form_submit_button(":material/refresh:")
    
    if fetch_model_info_clicked:
        with st.spinner("Getting model info"):
            try:
                model_info = None
                model_info = get_uc_model_info(uc_model_name, uc_model_version)
                st.session_state["import_uc_model_info"] = model_info
            except Exception as e:                    
                model_info_error = True
                del st.session_state["import_uc_model_info"]
    
    if(model_info_error):
        st.error("Error fetching model details.")        
        if error_callback:
            error_callback()
            
    if model_info:
        with st.form("import_model_uc_form_import", enter_to_submit=False):
            model_name = st.text_input("Model Name:", value=uc_model_name.split(".")[2], help="Common name of the mode if different from UC name.")
            model_source_version = st.text_input("Source Model Version:" , help="Source version of the corresponding UC model.")
            model_display_name = st.text_input("Display Name:",value=uc_model_name.split(".")[2], help="Name that will be displayed on UI.")
            model_description_url = st.text_input("Description URL:", help="A website URL where users can read more about this model")
            
            if st.form_submit_button('Import Model'):
                uc_import_model_clicked = True

    if uc_import_model_clicked:
        with st.spinner("Importing model"):
            try:
                import_model_from_uc(user_email = user_info.user_email,
                    model_category = model_category,
                    model_uc_name = uc_model_name,
                    model_uc_version =  uc_model_version, 
                    model_name = model_name,
                    model_source_version = model_source_version,
                    model_display_name = model_display_name,
                    model_description_url = model_description_url)
                
                model_info = None
                del st.session_state["import_uc_model_info"]
            except Exception as e:
                model_import_error = True

    if uc_import_model_clicked:
        if model_import_error:
            st.error("Error importing model") 
            if error_callback:
                error_callback()
        else:
            st.success("Model Imported Successfully.")
            if success_callback:
                success_callback()
            #if st.button("Close"):
            #    if "import_button" in st.session_state:
            #        del st.session_state["import_button"]              
            #    st.rerun()