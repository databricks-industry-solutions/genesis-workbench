"""Tiny helpers for building human-clickable links into the Databricks
workspace UI. Used by the View-in-Databricks anchors on dispatch banners
and the run-name links in search-past-runs tables."""
from __future__ import annotations

import os


def workspace_host() -> str:
    """Workspace URL. Streamlit reads `DATABRICKS_HOSTNAME`; the Apps SDK
    also exposes `DATABRICKS_HOST` pointing at the same workspace URL.
    Either works."""
    host = (
        os.environ.get("DATABRICKS_HOSTNAME")
        or os.environ.get("DATABRICKS_HOST")
        or ""
    ).rstrip("/")
    if not host:
        return ""
    if not host.startswith("http"):
        host = "https://" + host
    return host


def job_run_url(job_id: str | int | None, job_run_id: str | int | None) -> str:
    """Modern Jobs UI run-page URL. Returns empty string when any required
    piece is missing — callers should guard the render."""
    if not job_id or not job_run_id:
        return ""
    host = workspace_host()
    if not host:
        return ""
    return f"{host}/jobs/{job_id}/runs/{job_run_id}"


def dashboard_embed_url(dashboard_id: str | None, params: dict[str, str] | None = None) -> str:
    """Lakeview (dashboardsv3) embed URL. Optional `params` are passed as
    raw `?<keyword>=<value>` query string entries — Lakeview matches each
    against the dashboard parameter declared with that `keyword` and binds
    it into the SQL queries' `:name` placeholders."""
    if not dashboard_id:
        return ""
    host = workspace_host()
    if not host:
        return ""
    url = f"{host}/embed/dashboardsv3/{dashboard_id}"
    if params:
        from urllib.parse import urlencode
        url += "?" + urlencode({k: v for k, v in params.items() if v})
    return url
