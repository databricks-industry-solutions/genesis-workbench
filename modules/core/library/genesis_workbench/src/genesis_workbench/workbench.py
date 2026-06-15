
from databricks.sdk.core import Config, oauth_service_principal
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState, JobAccessControlRequest, JobPermissionLevel
from databricks.sdk.service.serving import ServingEndpointAccessControlRequest, ServingEndpointPermissionLevel

from databricks import sql

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List

@dataclass
class AppContext:
    core_catalog_name : str
    core_schema_name: str

@dataclass
class UserInfo:
    user_email : str
    user_name: str
    user_id: str
    user_groups: List[str]
    user_access_token:str
    user_display_name : str

@dataclass
class WarehouseInfo:
    id:str
    name: str
    cluster_size: str
    hostname: str
    http_path: str


def credential_provider():
    config = Config(
        host          = os.getenv("DATABRICKS_HOSTNAME"),
        client_id     = os.getenv("DATABRICKS_CLIENT_ID"),
        client_secret = os.getenv("DATABRICKS_CLIENT_SECRET"))
    return oauth_service_principal(config)

def get_warehouse_details_from_id(warehouse_id) -> WarehouseInfo:
    w = WorkspaceClient()
    details = w.warehouses.get(warehouse_id)
    return WarehouseInfo(
        id = details.id,
        name = details.name,
        cluster_size = details.cluster_size,        
        hostname = details.odbc_params.hostname,
        http_path = details.odbc_params.path
    )

def db_connect():
    warehouse_id = os.getenv("SQL_WAREHOUSE")
    warehouse_details = get_warehouse_details_from_id(warehouse_id)

    #print(f"SQL Warehouse Id: {warehouse_id}")

    os.environ["DATABRICKS_HOSTNAME"] = warehouse_details.hostname

    if os.getenv("IS_TOKEN_AUTH","")=="Y":
        #print(f"Connecting to  warehouse: {warehouse_details.http_path}, warehouse host: {warehouse_details.hostname} using a token")

        return sql.connect(server_hostname = warehouse_details.hostname,
            http_path = warehouse_details.http_path,
            access_token = os.getenv("DATABRICKS_TOKEN"))
    else:
        #print(f"Connecting to warehouse: {warehouse_details.http_path}, warehouse host: {warehouse_details.hostname} using oauth")

        return sql.connect(server_hostname = warehouse_details.hostname,
            http_path = warehouse_details.http_path,
            credentials_provider = credential_provider)


def execute_select_query(query)-> pd.DataFrame:
    with(db_connect()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [ col_desc[0] for col_desc in cursor.description]
            result = pd.DataFrame.from_records(cursor.fetchall(),columns=columns)
            return result

def execute_parameterized_inserts(param_query : str, list_of_records:list[list]) :
    with(db_connect()) as connection:
        with connection.cursor() as cursor:
            cursor.executemany(param_query, list_of_records )

def execute_non_select_query(query):
    with(db_connect()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)            

def execute_workflow(job_id: int, params: dict) -> str:
    w = WorkspaceClient()
    print(f"Running job with ID : {job_id}")
    run = w.jobs.run_now(
        job_id=job_id,
        job_parameters=params
    )
    return run.run_id

def get_workflow_job_status(
    tag_key: str = "application",
    tag_value: str = "genesis_workbench",
    days_back: int = 7,
    creator_filter: str = None
) -> dict:
    """
    Fetch workflow job runs filtered by tag, recent N days, and creator name.
    """
    w = WorkspaceClient()
    result = {}

    # Calculate cutoff timestamp in milliseconds
    cutoff_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)

    try:
        for job in w.jobs.list():
            tags = getattr(job.settings, "tags", {}) or {}
            if tags.get(tag_key) == tag_value:
                job_info = {
                    "job_id": job.job_id,
                    "name": job.settings.name,
                    "tags": tags,
                    "runs": []
                }

                try:
                    # We do not limit by number of runs; we filter by date instead
                    for run in w.jobs.list_runs(job_id=job.job_id):
                        if run.start_time and run.start_time >= cutoff_time:
                            if creator_filter and run.creator_user_name != creator_filter:
                                continue  # Skip runs not matching the creator

                            run_info = {
                                "run_id": run.run_id,
                                "state": run.state.life_cycle_state,
                                "result_state": run.state.result_state,
                                "start_time": run.start_time,
                                "end_time": run.end_time,
                                "creator_user_name": run.creator_user_name,
                            }
                            job_info["runs"].append(run_info)
                except DatabricksError as e:
                    print(f"Error retrieving runs for job ID {job.job_id}: {e}")

                result[job.settings.name] = job_info

    except DatabricksError as e:
        print(f"Error listing jobs: {e}")

    return result

def wait_for_job_run_completion(run_id: int, timeout: int = 600, poll_interval: int = 10):
    """
    Waits for a Databricks job run to complete.

    Args:
        w: An instance of WorkspaceClient (authenticated).
        run_id: The Databricks job run ID to monitor.
        timeout: The maximum time (in seconds) to wait for completion.
        poll_interval: How often (in seconds) to poll the run status.

    Raises:
        TimeoutError: If the job does not complete within the timeout.
        Exception: For other unexpected job run statuses.

    Returns:
        The final run object as returned by get_run.
    """
    w = WorkspaceClient()
    start = time.time()
    while True:
        run_info = w.jobs.get_run(run_id)
        life_cycle = run_info.state.life_cycle_state
        result_state = run_info.state.result_state

        print(str(life_cycle))
        
        print(f"Job run {run_id}: Life cycle state = {life_cycle}, Result state = {result_state}")

        if life_cycle in [RunLifeCycleState.TERMINATED, RunLifeCycleState.SKIPPED, RunLifeCycleState.INTERNAL_ERROR]:
            if result_state and result_state == RunResultState.SUCCESS:
                print(f"Job run {run_id} succeeded.")
                return run_info
            else:
                raise Exception(f"Job run {run_id} failed with result: {result_state}")

        if time.time() - start > timeout:
            raise TimeoutError(f"Job run {run_id} did not complete within {timeout} seconds.")

        time.sleep(poll_interval)

def get_deployed_modules():
    """ Get deployed modules from model table"""
    
    return_results = []
    core_catalog_name = os.environ["CORE_CATALOG_NAME"]
    core_schema_name = os.environ["CORE_SCHEMA_NAME"]
    query = f"""
        SELECT DISTINCT module FROM  {core_catalog_name}.{core_schema_name}.settings
    """
    result_df = execute_select_query(query)
    
    if len(result_df) > 0:
        return_results = result_df.iloc[:, 0].tolist()

    return return_results



def initialize(core_catalog_name:str, core_schema_name:str, sql_warehouse_id:str, token=None):
    """Initilializes the env variables required for workbench"""

    print("✳️ Initializing Genesis Workbench")

    os.environ["CORE_CATALOG_NAME"]=core_catalog_name
    os.environ["CORE_SCHEMA_NAME"]=core_schema_name
    os.environ["SQL_WAREHOUSE"]=sql_warehouse_id

    if token:
        os.environ["IS_TOKEN_AUTH"]="Y"
        os.environ["DATABRICKS_TOKEN"]=token

    query = f"SELECT * FROM {core_catalog_name}.{core_schema_name}.settings"
        
    result_df = execute_select_query(query)
    for idx, row in result_df.iterrows():
        os.environ[ str(row['key']).upper()] = str(row['value'])


def get_user_settings(user_email: str) -> dict:
    core_catalog_name = os.environ["CORE_CATALOG_NAME"]
    core_schema_name = os.environ["CORE_SCHEMA_NAME"]

    query = f"SELECT key,value FROM {core_catalog_name}.{core_schema_name}.user_settings WHERE user_email='{user_email}'"

    result_df = execute_select_query(query)
    
    if len(result_df) > 0:
        return dict(zip(result_df['key'], result_df['value']))
    else:
        return {}

def save_user_settings(user_email:str, user_settings:dict):
    core_catalog_name = os.environ["CORE_CATALOG_NAME"]
    core_schema_name = os.environ["CORE_SCHEMA_NAME"]
    
    print(f"Deleting existing user settings for {user_email}")
    delete_query=f"DELETE FROM {core_catalog_name}.{core_schema_name}.user_settings WHERE user_email='{user_email}'"
    execute_non_select_query(delete_query)

    print(f"Inserting new settings for {user_email}")
    insert_fields = ",".join([ f"('{user_email}','{k}','{v}')" for k,v in user_settings.items()])
    insert_query = f"INSERT INTO {core_catalog_name}.{core_schema_name}.user_settings (user_email, key, value) VALUES {insert_fields}"
    execute_non_select_query(insert_query)


def _list_app_names() -> List[str]:
    """Returns the apps that should receive CAN_MANAGE_RUN/CAN_QUERY grants
    when models/jobs are registered.

    Prefers DATABRICKS_APP_NAMES (comma-separated) for multi-app installs;
    falls back to DATABRICKS_APP_NAME for single-app installs (the default)."""
    raw = os.environ.get("DATABRICKS_APP_NAMES") or os.environ.get("DATABRICKS_APP_NAME", "")
    return [n.strip() for n in raw.split(",") if n.strip()]


def set_app_permissions_for_job(job_id:str, user_email:str):
    w = WorkspaceClient()
    acl = [
        JobAccessControlRequest(
            user_name=user_email,
            permission_level=JobPermissionLevel.IS_OWNER
        )
    ]
    for app_name in _list_app_names():
        try:
            app_sp = w.apps.get(name=app_name).service_principal_client_id
        except Exception as e:
            print(f"⚠️  Skipping job permission for app '{app_name}': {e}")
            continue
        acl.append(
            JobAccessControlRequest(
                user_name=app_sp,
                permission_level=JobPermissionLevel.CAN_MANAGE_RUN
            )
        )

    # ADDITIVE grant (PATCH semantics) — preserves any existing ACL rather than
    # replacing it. Critical for multi-app installs: a module re-registration
    # must NOT wipe a previously-granted sibling app (e.g. the MCP server SP,
    # granted by grant_app_permissions_job). update_permissions only adds the
    # owner + app grants listed here; set_permissions would clobber the rest.
    w.jobs.update_permissions(
        job_id=job_id,
        access_control_list=acl
    )


def set_app_permissions_for_volume(volume_full_name: str, write: bool = False):
    """Grant READ_VOLUME (and optionally WRITE_VOLUME) on a UC volume to
    every registered Databricks Apps service principal.

    Use this for any volume the application backend itself touches —
    e.g. uploads (motif.pdb for enzyme_optimization) or downloads
    (result PDBs, mapping JSONs). Jobs that read/write volumes still
    run as their owner; this is only needed when the app's SP issues
    files API calls directly.

    `volume_full_name` is the three-part UC name: 'catalog.schema.volume'.
    Idempotent — repeated calls converge to the same ACL.
    """
    w = WorkspaceClient()
    privileges = ["READ_VOLUME"]
    if write:
        privileges.append("WRITE_VOLUME")
    for app_name in _list_app_names():
        try:
            app_sp = w.apps.get(name=app_name).service_principal_client_id
        except Exception as e:
            print(f"⚠️  Skipping volume permission for app '{app_name}': {e}")
            continue
        try:
            # The SDK's grants.update takes SecurableType + full name + a list
            # of changes (principal + add). Volume = SecurableType.VOLUME.
            from databricks.sdk.service.catalog import (
                Privilege,
                PermissionsChange,
                SecurableType,
            )
            w.grants.update(
                securable_type=SecurableType.VOLUME,
                full_name=volume_full_name,
                changes=[
                    PermissionsChange(
                        principal=app_sp,
                        add=[Privilege[p] for p in privileges],
                    )
                ],
            )
            print(f"Granted {privileges} on volume {volume_full_name} to app '{app_name}' (sp={app_sp}).")
        except Exception as e:
            print(f"⚠️  Failed to grant {privileges} on volume {volume_full_name} for app '{app_name}': {e}")


def set_app_permissions_for_endpoint(endpoint_name:str):
    w = WorkspaceClient()
    endpoint_id = w.serving_endpoints.get(endpoint_name).id

    acl: List[ServingEndpointAccessControlRequest] = []
    for app_name in _list_app_names():
        try:
            app_sp = w.apps.get(name=app_name).service_principal_client_id
        except Exception as e:
            print(f"⚠️  Skipping endpoint permission for app '{app_name}': {e}")
            continue
        acl.append(
            ServingEndpointAccessControlRequest(
                user_name=app_sp,
                permission_level=ServingEndpointPermissionLevel.CAN_QUERY
            )
        )

    # Replaces the existing ACL with the listed apps' grants.
    w.serving_endpoints.update_permissions(
        serving_endpoint_id=endpoint_id,
        access_control_list=acl
    )

