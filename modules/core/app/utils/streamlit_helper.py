import streamlit as st
from genesis_workbench.workbench import UserInfo, AppContext
from databricks.sdk import WorkspaceClient

def get_user_info():
    headers = st.context.headers
    user_access_token = headers.get("X-Forwarded-Access-Token")
    user_name=headers.get("X-Forwarded-Preferred-Username")
    user_display_name = ""
    if user_access_token:
        # Initialize WorkspaceClient with the user's token
        w = WorkspaceClient(token=user_access_token, auth_type="pat")
        # Get current user information
        current_user = w.current_user.me()
        # Display user information
        user_display_name = current_user.display_name

    return UserInfo(
        user_name=user_name,
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
        user_access_token = headers.get("X-Forwarded-Access-Token"),
        user_display_name = user_display_name if user_display_name != "" else user_name
    )

def get_app_context() -> AppContext:
    context_str = ""
    with open("extra_params.txt", "r") as file:
        context_str = file.read()

    #context str is like --var="dev_user_prefix=scn,core_catalog_name=genesis_workbench,core_schema_name=dev_srijit_nair_dbx_genesis_workbench_core"

    context_str = context_str.replace("--var=","").replace("\"","")

    ctx_items = {}
    [(lambda x : ctx_items.update({x[0]:x[1]}) )(ctx_item.split("=")) for ctx_item in context_str.split(",")] 
    
    appContext = AppContext(
        core_catalog_name=ctx_items["core_catalog_name"],
        core_schema_name=ctx_items["core_schema_name"]
    )

    return appContext