
from databricks.sdk.core import Config, oauth_service_principal
from databricks import sql
import os

def credential_provider(sql_warehouse_hostname):
  config = Config(
    host          = f"https://{sql_warehouse_hostname}",
    client_id     = os.getenv("DATABRICKS_CLIENT_ID"),
    client_secret = os.getenv("DATABRICKS_CLIENT_SECRET"))
  return oauth_service_principal(config)

def db_connect(sql_warehouse_hostname, http_path):
    return sql.connect(server_hostname = sql_warehouse_hostname,
        http_path = http_path,
        credentials_provider = credential_provider)


