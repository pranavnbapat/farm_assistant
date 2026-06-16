#!/bin/sh
# Pull the latest image and recreate the farm_assistant arena containers.
#
# Resilient by design: a missing .env file, an un-pullable image, or a single
# failing service will NOT abort the rest — the script attempts every service,
# reports what was skipped, and exits 0. Re-running is safe (idempotent: only
# containers whose image/config changed are recreated).
#
# Usage:
#   ./pull_and_restart.sh                      # update all arena services
#   ./pull_and_restart.sh farm_assistant       # update just one
#   SERVICES="a b" ./pull_and_restart.sh       # or via env
#   COMPOSE_FILE=docker-compose.yml ./pull_and_restart.sh

set -u

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

# --- locate a working compose command -------------------------------------
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DC="docker-compose"
else
    echo "ERROR: Docker Compose is not installed." >&2
    exit 1
fi

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "ERROR: compose file '$COMPOSE_FILE' not found in $(pwd)." >&2
    exit 1
fi
DC="$DC -f $COMPOSE_FILE"

# --- decide which services to update --------------------------------------
# Priority: CLI args > $SERVICES env > services discovered from the compose
# file > a built-in default list (used when discovery fails, e.g. because one
# service references a missing env_file).
SERVICES="${*:-${SERVICES:-}}"
if [ -z "$SERVICES" ]; then
    SERVICES="$($DC config --services 2>/dev/null || true)"
fi
if [ -z "$SERVICES" ]; then
    echo "WARN: could not list services from '$COMPOSE_FILE'" >&2
    echo "      (a missing env_file can cause this); using built-in default list." >&2
    SERVICES="farm_assistant farm_assistant_openai farm_assistant_anthropic farm_assistant_eurollm"
fi

echo "Compose file : $COMPOSE_FILE"
echo "Services     : $(echo $SERVICES | tr '\n' ' ')"
echo

# --- pull (ignore individual image failures) ------------------------------
echo "== Pulling images =="
# shellcheck disable=SC2086
$DC pull --ignore-pull-failures $SERVICES \
    || echo "WARN: one or more images failed to pull; continuing with what is available." >&2
echo

# --- recreate each service independently ----------------------------------
echo "== Recreating containers =="
skipped=""
for svc in $SERVICES; do
    echo "-- $svc --"
    if $DC up -d "$svc"; then
        :
    else
        echo "WARN: '$svc' did not start (missing .env, undefined service, or bad config?). Skipping." >&2
        skipped="$skipped $svc"
    fi
done
echo

# --- status ----------------------------------------------------------------
echo "== Status =="
$DC ps || true
echo

if [ -n "$skipped" ]; then
    echo "Done with warnings. Skipped:$skipped"
else
    echo "Done — all services updated."
fi

# Never fail the deploy on a partial problem; warnings above say what to fix.
exit 0
