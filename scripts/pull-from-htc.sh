#!/usr/bin/env bash
#
# Download files or directories from dbansa11@sol.asu.edu to local.
# Uses rsync over SSH.
#
# Usage:
#   ./scripts/pull-from-htc.sh remote_path [local_path]
#   ./scripts/pull-from-htc.sh user@host:remote_path [local_path]   # override host
#
# Examples:
#   ./scripts/pull-from-htc.sh ~/runs ./runs
#   ./scripts/pull-from-htc.sh ~/outputs/model.pt .
#
set -e

REMOTE_DEFAULT="dbansa11@sol.asu.edu"
REMOTE_SPEC="$1"
LOCAL_DEST="${2:-.}"

if [[ -z "$REMOTE_SPEC" ]]; then
  echo "Usage: $0 remote_path [local_path]" >&2
  echo "       (remote is on $REMOTE_DEFAULT unless you pass user@host:path)" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  $0 ~/runs ./runs" >&2
  echo "  $0 ~/outputs/model.pt ." >&2
  exit 1
fi

# If no ':' in first arg, treat as path on default host
if [[ "$REMOTE_SPEC" != *:* ]]; then
  REMOTE_SPEC="${REMOTE_DEFAULT}:${REMOTE_SPEC}"
fi

echo "Pulling: $REMOTE_SPEC -> $LOCAL_DEST"
exec rsync -avz --progress -e ssh "$REMOTE_SPEC" "$LOCAL_DEST"
