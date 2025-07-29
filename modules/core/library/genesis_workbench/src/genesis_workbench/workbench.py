
from databricks.sdk.core import Config, oauth_service_principal
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError

from databricks import sql

import os
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class AppContext:
    core_catalog_name : str
    core_schema_name: str

@dataclass
class UserInfo:
    user_email : str
    user_name: str
    user_id: str
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

    print(f"SQL Warehouse Id: {warehouse_id}")

    os.environ["DATABRICKS_HOSTNAME"] = warehouse_details.hostname

    if os.getenv("IS_TOKEN_AUTH","")=="Y":
        print(f"Connecting to  warehouse: {warehouse_details.http_path}, warehouse host: {warehouse_details.hostname} using a token")

        return sql.connect(server_hostname = warehouse_details.hostname,
            http_path = warehouse_details.http_path,
            access_token = os.getenv("DATABRICKS_TOKEN"))
    else:
        print(f"Connecting to warehouse: {warehouse_details.http_path}, warehouse host: {warehouse_details.hostname} using oauth")

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

def execute_insert_delete_query(query):
    with(db_connect()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)            

def execute_workflow(job_id: int, params: dict) -> str:
    w = WorkspaceClient()
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

def get_workbench_settings() -> dict:
    """Method to application settings like job id etc"""

    query = f"SELECT * FROM \
                {os.environ['CORE_CATALOG_NAME']}.{os.environ['CORE_SCHEMA_NAME']}.settings"
        
    result_df = execute_select_query(query)

    model_info = result_df.apply(lambda row: GWBModelInfo(**row), axis=1).tolist()[0]
    return model_info    
    