#!/usr/bin/env bash
#
# Sync runs/ and logs/ from a project directory on dbansa11@sol.asu.edu to local.
# Uses rsync over SSH. Creates local runs and logs dirs if they don't exist.
#
# Usage:
#   ./scripts/pull-from-htc.sh remote_project_path [local_dest]
#   ./scripts/pull-from-htc.sh user@host:remote_project_path [local_dest]   # override host
#
# Examples:
#   ./scripts/pull-from-htc.sh ~/capstone
#   ./scripts/pull-from-htc.sh ~/capstone .
#   ./scripts/pull-from-htc.sh ~/capstone ./backup
#
set -e

REMOTE_DEFAULT="dbansa11@sol.asu.edu"
REMOTE_SPEC="$1"
LOCAL_DEST="${2:-.}"

if [[ -z "$REMOTE_SPEC" ]]; then
  echo "Usage: $0 remote_project_path [local_dest]" >&2
  echo "       Syncs runs/ and logs/ from remote project into local_dest (default: .)" >&2
  echo "       Remote is on $REMOTE_DEFAULT unless you pass user@host:path" >&2
  echo "" >&2
  exit 1
fi

# If no ':' in first arg, treat as path on default host
if [[ "$REMOTE_SPEC" != *:* ]]; then
  REMOTE_SPEC="${REMOTE_DEFAULT}:${REMOTE_SPEC}"
fi

mkdir -p "$LOCAL_DEST/runs" "$LOCAL_DEST/logs"

REMOTE_RUNS="${REMOTE_SPEC}/runs"
REMOTE_LOGS="${REMOTE_SPEC}/logs"

echo "Pulling runs: $REMOTE_RUNS -> $LOCAL_DEST/runs"
rsync -avz --progress -e ssh "$REMOTE_RUNS/" "$LOCAL_DEST/runs/"

echo "Pulling logs: $REMOTE_LOGS -> $LOCAL_DEST/logs"
rsync -avz --progress -e ssh "$REMOTE_LOGS/" "$LOCAL_DEST/logs/"

echo "Done."
