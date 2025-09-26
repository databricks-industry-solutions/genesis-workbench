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
    #components.html("<script>parent.window.location.reload()</script>")
    st.switch_page("views/home.py")


st.title(":material/account_circle: Profile")

with st.spinner("Getting user information"):
    user_info = get_user_info()
    if "user_settings" not in st.session_state:        
        user_settings = get_user_settings(user_email=user_info.user_email)
        st.session_state["user_settings"] = user_settings
    
    user_settings = st.session_state["user_settings"]

with st.form("user_profile_setup_form", enter_to_submit=False):
    
    st.markdown("### General")
    st.text_input("Email:", user_info.user_email, disabled=True)
    display_name = st.text_input("Display Name:", user_settings["user_display_name"] if "user_display_name" in user_settings else user_info.user_display_name) 

    st.divider()

    st.markdown("### MLflow Setup")
    st.write("Genesis Workbench may create experiments to track your work and its recommended to create them inside your workspace folder.")
    st.write(f"#### :material/keyboard_double_arrow_right: Step 1")    
    st.write(f"##### Create a new folder in `/Workspace/Users/{user_info.user_email}/` ")
    cc1,cc2 = st.columns([1,1])
    with cc1:
        st.image("images/demo_new_folder_small.gif" )

    mlflow_experiment_folder = st.text_input(f"Enter the folder name:","mlflow_experiments")

    st.write(f"#### :material/keyboard_double_arrow_right: Step 2")    
    st.write(f"##### Grant Permission to the application service principal: `{ os.environ['DATABRICKS_CLIENT_ID'] if 'DATABRICKS_CLIENT_ID' in os.environ else 'none' }` ")
    ac1,ac2 = st.columns([2,1])
    with ac1:
        st.image("images/set_permissions_small.gif" )
    st.write(f"- Navigate to the above workspace folder and click the `Share` button on top right corner.")
    st.write(f"- Grant `Can Manage` permission to the above service principal id")
    st.write(f"- Click the `Check Folder Permission and Save` button below")

    profile_save_button = st.form_submit_button('Check Folder Permission and Save')


if profile_save_button:
    mlflow_check_success = False
    try:
        with st.spinner(f"Checking permissions on /Workspace/Users/{user_info.user_email}/{mlflow_experiment_folder}"):
            test_mlflow_experiment(user_info.user_email, base_folder=mlflow_experiment_folder)
            mlflow_check_success = True
            st.success("Experiment access verified.")
    except Exception as e:
        st.error("Experiment folder access failed. Make sure the folder exists and access is granted to the application service principal.")

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
            time.sleep(1)                    
            reload()

