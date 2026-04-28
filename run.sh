#!/bin/bash
set -euo pipefail

# Farm Assistant Run Script
# Preferred usage:
#   ./run.sh --mode local --backend dev
#   ./run.sh --mode local --backend prd
#   ./run.sh --mode local
#   ./run.sh --mode docker
#
# Backward-compatible shortcuts:
#   ./run.sh local dev
#   ./run.sh local prd
#   ./run.sh local
#   ./run.sh docker
#   ./run.sh dev
#   ./run.sh prd

MODE="local"
BACKEND_PROFILE="local"

DEV_BACKEND_BASE="https://backend-admin.dev.farmbook.ugent.be"
PRD_BACKEND_BASE="https://backend-admin.prd.farmbook.ugent.be"

usage() {
    cat <<'EOF'
Usage:
  ./run.sh --mode local --backend [local|dev|prd]
  ./run.sh --mode docker

Shortcuts:
  ./run.sh local [local|dev|prd]
  ./run.sh docker
  ./run.sh dev
  ./run.sh prd

Examples:
  ./run.sh --mode local --backend dev
  ./run.sh --mode local --backend prd
  ./run.sh local dev
  ./run.sh prd
  ./run.sh docker
EOF
}

parse_args() {
    if [ "$#" -eq 0 ]; then
        return
    fi

    case "${1:-}" in
        dev|prd)
            MODE="local"
            BACKEND_PROFILE="$1"
            return
            ;;
        local|docker)
            MODE="$1"
            if [ "$#" -ge 2 ]; then
                BACKEND_PROFILE="$2"
            fi
            return
            ;;
    esac

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --mode)
                [ "$#" -ge 2 ] || { echo "Missing value for --mode"; usage; exit 1; }
                MODE="$2"
                shift 2
                ;;
            --backend)
                [ "$#" -ge 2 ] || { echo "Missing value for --backend"; usage; exit 1; }
                BACKEND_PROFILE="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown argument: $1"
                usage
                exit 1
                ;;
        esac
    done
}

parse_args "$@"

if [ "$MODE" == "docker" ]; then
    echo "Starting Farm Assistant with Docker Compose..."
    docker-compose up --build
elif [ "$MODE" == "local" ]; then
    echo "Starting Farm Assistant locally with Uvicorn..."

    # Check if virtual environment exists
    if [ -d ".venv" ]; then
        echo "Activating virtual environment..."
        source .venv/bin/activate
    fi

    case "$BACKEND_PROFILE" in
        local)
            echo "Using local backend profile from .env"
            ;;
        dev)
            export FA_ENV=dev
            export AUTH_BACKEND_URL="$DEV_BACKEND_BASE"
            export CHAT_BACKEND_URL="$DEV_BACKEND_BASE"
            echo "Using dev backend profile:"
            echo "  AUTH_BACKEND_URL=$AUTH_BACKEND_URL"
            echo "  CHAT_BACKEND_URL=$CHAT_BACKEND_URL"
            ;;
        prd)
            export FA_ENV=prd
            export AUTH_BACKEND_URL="$PRD_BACKEND_BASE"
            export CHAT_BACKEND_URL="$PRD_BACKEND_BASE"
            echo "Using prd backend profile:"
            echo "  AUTH_BACKEND_URL=$AUTH_BACKEND_URL"
            echo "  CHAT_BACKEND_URL=$CHAT_BACKEND_URL"
            ;;
        *)
            echo "Invalid backend profile: $BACKEND_PROFILE"
            usage
            exit 1
            ;;
    esac

    # Run the application
    uvicorn app.main:app --reload --host 0.0.0.0 --port 18000 --timeout-keep-alive 120
else
    echo "Invalid mode: $MODE"
    usage
    exit 1
fi
