import streamlit as st
import os
import base64
import mlflow
import time
import streamlit.components.v1 as components
from utils.streamlit_helper import get_user_info  
from databricks.sdk import WorkspaceClient
from genesis_workbench.workbench import get_user_settings, save_user_settings

def test_mlflow_experiment(user_email, base_folder, ):     
    w = WorkspaceClient()

    mlflow_experiment_base_path = f"Users/{user_email}/{base_folder}"
    w.workspace.mkdirs(f"/Workspace/{mlflow_experiment_base_path}")
    experiment_path = f"/{mlflow_experiment_base_path}/__test__"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.set_experiment(experiment_path)
    print("Deleting the experiment __test__")
    mlflow.delete_experiment(experiment_id= experiment.experiment_id)

def reload():
    components.html("<script>parent.window.location.reload()</script>")

st.title(":material/account_circle: Profile")

with st.spinner("Getting user information"):
    user_info = get_user_info()
    if "user_settings" not in st.session_state:        
        user_settings = get_user_settings(user_email=user_info.user_email)
        st.session_state["user_settings"] = user_settings
    
    user_settings = st.session_state["user_settings"]

with st.form("user_profile_setup_form", enter_to_submit=False):
    st.text_input("Email:", user_info.user_email, disabled=True)
    display_name = st.text_input("Display Name:", user_settings["user_display_name"] if "user_display_name" in user_settings else user_info.user_display_name) 

    st.markdown("#### Setup MLflow Experiment Location")
    st.markdown("###### Create/Identify Folder")
    st.write("Genesis Workbench might create experiments and log runs to those experiments. \
            These experiments will be created in a workspace folder you specify. Please perfom the below steps.")
    
    st.write("- If you already have a folder that you would like to use, please specify it below.")

    st.write("- If you dont, please create a new folder in your workspace folder and specify the folder name below.")

    mlflow_experiment_folder = st.text_input(f"MLflow experiment folder inside : /Workspace/Users/{user_info.user_email}/","mlflow_experiments")

    st.markdown("###### Grant Permission")
    st.write("Now you need to share the folder with the application so that Genesis Workbench has permission to create experiments/runs.")

    st.write("Step 1: Navigate to the above workspace folder and click the `Share` button on top right corner.")
    
    st.write(f"Step 2: Grant `Can Manage` permission to this service principal: `{ os.environ['DATABRICKS_CLIENT_ID'] if 'DATABRICKS_CLIENT_ID' in os.environ else 'none' }` ")

    profile_save_button = st.form_submit_button('Check Folder Permission and Save')


if profile_save_button:
    mlflow_check_success = False
    try:
        with st.spinner(f"Checking permissions on /Workspace/Users/{user_info.user_email}/{mlflow_experiment_folder}"):
            test_mlflow_experiment(user_info.user_email, base_folder=mlflow_experiment_folder)
            mlflow_check_success = True
            st.success("Experiment access verified.")
    except Exception as e:
        st.error("Experiment folder access failed.")

    if mlflow_check_success:
        try:
            with st.spinner("Saving settings"):
                save_user_settings(user_email=user_info.user_email, user_settings={
                    "user_display_name" : display_name,
                    "mlflow_experiment_folder" : mlflow_experiment_folder,
                    "setup_done" : "Y"
                })

                user_settings = get_user_settings(user_email=user_info.user_email)
                st.session_state["user_settings"] = user_settings
                st.success("Settings Saved.")

        except Exception as e:
            st.error("Save Failed. Try Again")

        with st.spinner("Reloading"):
            time.sleep(3)                    
            reload()

