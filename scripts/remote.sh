#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

: "${REMOTE_HOST:?Set REMOTE_HOST in .env}"
: "${REMOTE_USER:?Set REMOTE_USER in .env}"
: "${REMOTE_PASSWORD:?Set REMOTE_PASSWORD in .env}"

SOCKET="$HOME/.ssh/sockets/capstone-remote"
mkdir -p "$(dirname "$SOCKET")"

ssh_opts=(-o ControlPath="$SOCKET" -o ControlMaster=auto -o StrictHostKeyChecking=accept-new)

usage() {
    cat <<EOF
Usage: $0 <command> [args]

Commands:
    open                Open SSH connection (Duo push + password)
    run <cmd>           Run a command on the remote server
    sync <src> [dst]    Rsync files from remote to local (dst defaults to .)
    close               Close the SSH connection
    status              Check if connection is active
EOF
}

cmd_open() {
    if ssh "${ssh_opts[@]}" -O check "$REMOTE_USER@$REMOTE_HOST" 2>/dev/null; then
        echo "already connected"
        return 0
    fi

    if ! command -v expect &>/dev/null; then
        echo "error: 'expect' is required (brew install expect / apt install expect)" >&2
        exit 1
    fi

    echo "Approve the Duo push on your phone..."
    export REMOTE_PASSWORD REMOTE_USER REMOTE_HOST SOCKET

    # Sol flow: Duo autopushes first, you approve on phone, then password prompt appears.
    expect << 'EXPECT'
set timeout 120
spawn ssh -o ControlMaster=yes -o ControlPersist=yes \
    -o ControlPath=$env(SOCKET) \
    -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=60 \
    $env(REMOTE_USER)@$env(REMOTE_HOST) echo "AUTH_OK"

expect {
    -re "(?i)password" {
        send "$env(REMOTE_PASSWORD)\r"
        exp_continue
    }
    "AUTH_OK" {}
    timeout {
        send_error "Timed out — approve Duo push on your phone\n"
        exit 1
    }
    eof {
        send_error "Connection closed unexpectedly\n"
        exit 1
    }
}
EXPECT

    echo "connected to $REMOTE_HOST"
}

cmd_run() {
    if [[ $# -eq 0 ]]; then
        echo "usage: $0 run <command>" >&2
        exit 1
    fi
    ssh "${ssh_opts[@]}" "$REMOTE_USER@$REMOTE_HOST" "$@"
}

cmd_sync() {
    if [[ $# -eq 0 ]]; then
        echo "usage: $0 sync <remote_path> [local_path]" >&2
        exit 1
    fi
    local src="$1"
    local dst="${2:-.}"
    rsync -avz -e "ssh -o ControlPath=$SOCKET -o ControlMaster=auto" \
        "$REMOTE_USER@$REMOTE_HOST:$src" "$dst"
}

cmd_close() {
    ssh "${ssh_opts[@]}" -O exit "$REMOTE_USER@$REMOTE_HOST" 2>/dev/null || true
    echo "closed"
}

cmd_status() {
    if ssh "${ssh_opts[@]}" -O check "$REMOTE_USER@$REMOTE_HOST" 2>/dev/null; then
        echo "connected"
    else
        echo "not connected"
        return 1
    fi
}

case "${1:-}" in
    open)   cmd_open ;;
    run)    shift; cmd_run "$@" ;;
    sync)   shift; cmd_sync "$@" ;;
    close)  cmd_close ;;
    status) cmd_status ;;
    *)      usage; exit 1 ;;
esac
