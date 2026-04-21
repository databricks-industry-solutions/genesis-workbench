#!/usr/bin/env python3
"""
watch_gwb_jobs.py — live monitor for Genesis Workbench job runs.

Polls all Databricks jobs tagged `application=genesis_workbench` for a given
user + workspace (via CLI profile). Surfaces failures as they happen with the
underlying error trace so you don't wait for email-on-failure notifications.

Usage:
    python3 scripts/watch_gwb_jobs.py --profile <profile> [options]

Options:
    --profile   Databricks CLI profile (required)
    --interval  Seconds between polls (default 30; ignored with --once)
    --once      Take a single snapshot and exit (no polling)
    --module    Filter to one module (e.g., bionemo, disease_biology)
    --user      Filter to jobs created by a specific user email prefix
                (default: current CLI user)
    --verbose   Show full error traces instead of one-line summaries
    --max-iterations  Bound the polling loop (safety; default 240 = 2h @ 30s)

Exit codes:
    0 — all jobs in SUCCESS / NO-RUNS-YET state
    1 — any job in FAILED / TIMEDOUT / CANCELED state
    2 — CLI / auth error
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone


def run_cli(args, profile, parse_json=True):
    """Wrapper for databricks CLI calls with the given profile."""
    cmd = ["databricks"] + args + ["--profile", profile]
    if parse_json:
        cmd += ["--output", "json"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        return None
    except FileNotFoundError:
        print("ERROR: `databricks` CLI not on PATH. Install per docs.databricks.com.", file=sys.stderr)
        sys.exit(2)
    if not parse_json:
        return out.decode()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def get_current_user_prefix(profile):
    """Return the email prefix for the current authenticated user."""
    me = run_cli(["current-user", "me"], profile)
    if not me:
        return None
    emails = me.get("emails", [])
    if emails:
        return emails[0].get("value", "").split("@")[0] + "@"
    return None


def discover_gwb_jobs(profile, user_prefix=None, module_filter=None):
    """List GWB-tagged jobs on the workspace filtered by creator + optional module."""
    jobs = run_cli(["jobs", "list"], profile)
    if jobs is None:
        print("ERROR: Could not list jobs. Check auth: `databricks current-user me`.", file=sys.stderr)
        sys.exit(2)
    gwb = []
    for j in jobs:
        tags = j.get("settings", {}).get("tags", {})
        if tags.get("application") != "genesis_workbench":
            continue
        creator = j.get("creator_user_name", "")
        if user_prefix and not creator.startswith(user_prefix):
            continue
        mod = tags.get("module", "?")
        if module_filter and mod != module_filter:
            continue
        gwb.append({
            "job_id": j["job_id"],
            "name": j["settings"].get("name", "?"),
            "module": mod,
        })
    return gwb


def get_latest_run_state(profile, job_id):
    """Return the latest run's state for a given job, or None if no runs."""
    runs = run_cli(["jobs", "list-runs", f"--job-id={job_id}", "--limit=1"], profile)
    if not runs:
        return None
    r = runs[0]
    s = r.get("state", {})
    start_ms = r.get("start_time", 0)
    return {
        "run_id": r.get("run_id"),
        "life": s.get("life_cycle_state", ""),
        "result": s.get("result_state", ""),
        "message": s.get("state_message", "")[:200],
        "start": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%H:%M:%S") if start_ms else "?",
    }


def classify(state):
    if state is None:
        return "no_run"
    if state["life"] in ("RUNNING", "PENDING", "QUEUED"):
        return "running"
    if state["result"] == "SUCCESS":
        return "success"
    if state["result"] in ("FAILED", "CANCELED", "TIMEDOUT"):
        return "failed"
    return "other"


def fetch_failure_details(profile, run_id, verbose=False):
    """Fetch task-level failure details for a failed parent run."""
    detail = run_cli(["jobs", "get-run", str(run_id)], profile)
    if not detail:
        return []
    findings = []
    for task in detail.get("tasks", []):
        ts = task.get("state", {}).get("result_state", "")
        if ts not in ("FAILED", "TIMEDOUT", "CANCELED"):
            continue
        trid = task["run_id"]
        tkey = task.get("task_key", "?")
        msg = task.get("state", {}).get("state_message", "")[:200]
        err = None
        trace = None
        out = run_cli(["jobs", "get-run-output", str(trid)], profile)
        if out:
            err = out.get("error", "")[:400] if out.get("error") else None
            trace = out.get("error_trace", "") if out.get("error_trace") else None
        findings.append({
            "task_key": tkey,
            "state": ts,
            "message": msg,
            "error": err,
            "trace": trace if verbose else None,
        })
    return findings


COLOR = {
    "success": "\033[32m",
    "failed": "\033[31m",
    "running": "\033[33m",
    "no_run": "\033[90m",
    "reset": "\033[0m",
}

ICON = {
    "success": "🟢",
    "failed": "🔴",
    "running": "🟡",
    "no_run": "⚪",
    "other": "❓",
}


def print_snapshot(jobs, states, previous_states, verbose, profile):
    """Print a formatted snapshot + surface new failures with error details."""
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    buckets = defaultdict(list)
    transitions = []

    for j in jobs:
        jid = j["job_id"]
        state = states.get(jid)
        cls = classify(state)
        buckets[cls].append((j, state))

        prev = previous_states.get(jid)
        prev_cls = classify(prev) if prev is not None else None
        if prev_cls and prev_cls != cls:
            transitions.append((j, prev_cls, cls, state))

    total = len(jobs)
    print(f"\n{'='*80}")
    print(f"[{now}] {total} GWB jobs | "
          f"🟢 {len(buckets['success'])} | 🟡 {len(buckets['running'])} | "
          f"🔴 {len(buckets['failed'])} | ⚪ {len(buckets['no_run'])}")
    print(f"{'='*80}")

    if transitions:
        print(f"\nTransitions since last poll ({len(transitions)}):")
        for j, prev, cur, state in transitions:
            prev_icon = ICON.get(prev, "·")
            cur_icon = ICON.get(cur, "·")
            print(f"  {prev_icon} → {cur_icon}  [{j['module']:<20}] {j['name']}")

    if buckets["failed"]:
        print(f"\n🔴 FAILURES ({len(buckets['failed'])}):")
        for j, state in buckets["failed"]:
            print(f"  [{j['module']:<20}] {j['name']:<45} | {state['result']} @ {state['start']}")
            if state.get("message"):
                print(f"    state_msg: {state['message'][:150]}")
            findings = fetch_failure_details(profile, state["run_id"], verbose=verbose)
            for f in findings:
                print(f"    ❌ task {f['task_key']}: {f['state']}")
                if f.get("error"):
                    print(f"       error: {f['error']}")
                if f.get("trace"):
                    print(f"       trace (last 5 lines):")
                    for line in f["trace"].strip().split("\n")[-5:]:
                        print(f"         {line[:150]}")

    if buckets["running"]:
        print(f"\n🟡 RUNNING ({len(buckets['running'])}):")
        for j, state in buckets["running"]:
            print(f"  [{j['module']:<20}] {j['name']:<45} | {state['life']} @ {state['start']}")

    if buckets["no_run"] and len(buckets["no_run"]) < 20:
        print(f"\n⚪ NO RUNS YET ({len(buckets['no_run'])}):")
        for j, _ in buckets["no_run"]:
            print(f"  [{j['module']:<20}] {j['name']}")


def all_terminal(states, jobs):
    for j in jobs:
        s = states.get(j["job_id"])
        if s is None:
            continue  # no_run is terminal-ish (won't auto-start)
        if s["life"] in ("RUNNING", "PENDING", "QUEUED"):
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="Databricks CLI profile name")
    ap.add_argument("--interval", type=int, default=30, help="Seconds between polls (default 30)")
    ap.add_argument("--once", action="store_true", help="Single snapshot, no polling")
    ap.add_argument("--module", help="Filter to one module (e.g., bionemo)")
    ap.add_argument("--user", help="Filter to creator email prefix (default: current user)")
    ap.add_argument("--verbose", action="store_true", help="Full error traces")
    ap.add_argument("--max-iterations", type=int, default=240, help="Polling loop bound (default 240)")
    args = ap.parse_args()

    user_prefix = args.user
    if not user_prefix:
        user_prefix = get_current_user_prefix(args.profile)
        if not user_prefix:
            print("ERROR: Could not determine current user. Specify --user.", file=sys.stderr)
            sys.exit(2)

    print(f"Watching GWB jobs on profile={args.profile}, user prefix={user_prefix}"
          + (f", module={args.module}" if args.module else ""))

    jobs = discover_gwb_jobs(args.profile, user_prefix=user_prefix, module_filter=args.module)
    print(f"Discovered {len(jobs)} GWB-tagged jobs.")
    if not jobs:
        print("No matching jobs found. Nothing to watch.")
        return 0

    previous_states = {}
    exit_code = 0

    iteration = 0
    while True:
        iteration += 1
        states = {j["job_id"]: get_latest_run_state(args.profile, j["job_id"]) for j in jobs}
        print_snapshot(jobs, states, previous_states, args.verbose, args.profile)

        # Determine exit code from current state
        any_failed = any(classify(s) == "failed" for s in states.values())
        if any_failed:
            exit_code = 1

        previous_states = states

        if args.once:
            break
        if all_terminal(states, jobs):
            print("\nAll jobs reached terminal states. Exiting.")
            break
        if iteration >= args.max_iterations:
            print(f"\nHit max-iterations ({args.max_iterations}). Exiting.")
            break
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
