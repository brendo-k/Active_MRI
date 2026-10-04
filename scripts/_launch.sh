#!/usr/bin/env bash
# Sourced by the training launch scripts. Starts a detached job and fails loudly
# if python is unavailable or the job dies during startup, instead of echoing a PID
# for a process that never ran.

STARTUP_CHECK_SECONDS="${STARTUP_CHECK_SECONDS:-20}"
STARTUP_LOG_LINES=20

launch_detached() {
    local log="$1"; shift

    if ! command -v python >/dev/null; then
        echo "ERROR: python not found on PATH; activate the project env first (source .venv/bin/activate)" >&2
        return 1
    fi

    nohup setsid "$@" > "$log" 2>&1 &
    local pid=$!

    sleep "$STARTUP_CHECK_SECONDS"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: job (PID $pid) exited within ${STARTUP_CHECK_SECONDS}s of launch; last lines of $log:" >&2
        tail -n "$STARTUP_LOG_LINES" "$log" >&2
        return 1
    fi

    echo "PID: $pid"
    echo "Log: $log"
}
