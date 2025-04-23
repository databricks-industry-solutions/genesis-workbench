
from databricks.sdk.core import Config, oauth_service_principal
from databricks.sdk import WorkspaceClient
from databricks import sql
import os
import pandas as pd
from dataclasses import dataclass

@dataclass
class AppContext:
    core_catalog_name : str
    core_schema_name: str

@dataclass
class UserInfo:
    user_email : str
    user_name: str

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
    os.environ["DATABRICKS_HOSTNAME"] = warehouse_details.hostname
    
    if os.getenv("IS_LOCAL_TEST","")=="Y":
        return sql.connect(server_hostname = warehouse_details.hostname,
            http_path = warehouse_details.http_path,
            access_token = os.getenv("DATABRICKS_TOKEN"))
    else:
        return sql.connect(server_hostname = warehouse_details.hostname,
            http_path = warehouse_details.http_path,
            credentials_provider = credential_provider)

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

def execute_select_query(query)-> pd.DataFrame:
    with(db_connect()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [ col_desc[0] for col_desc in cursor.description]
            result = pd.DataFrame.from_records(cursor.fetchall(),columns=columns)
            return result
        
def execute_upsert_delete_query(query):
    with(db_connect()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)            
