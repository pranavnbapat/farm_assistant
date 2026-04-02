#!/bin/bash

# Farm Assistant Run Script
# Usage: ./run.sh [local|docker] [local|dev|prd]

MODE=${1:-local}
BACKEND_PROFILE=${2:-local}

DEV_BACKEND_BASE="https://backend-admin.dev.farmbook.ugent.be"
PRD_BACKEND_BASE="https://backend-admin.prd.farmbook.ugent.be"

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
            echo "Usage: ./run.sh [local|docker] [local|dev|prd]"
            echo ""
            echo "Examples:"
            echo "  ./run.sh local        # local UI + backend values from .env"
            echo "  ./run.sh local dev    # local UI + dev Django backend"
            echo "  ./run.sh local prd    # local UI + prd Django backend"
            exit 1
            ;;
    esac

    # Run the application
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --timeout-keep-alive 120
else
    echo "Usage: ./run.sh [local|docker] [local|dev|prd]"
    echo ""
    echo "Options:"
    echo "  local   - Run with Uvicorn (development)"
    echo "  docker  - Run with Docker Compose"
    echo ""
    echo "Backend profile (only used with local):"
    echo "  local   - Use backend values from .env"
    echo "  dev     - Override to dev Django backend"
    echo "  prd     - Override to prd Django backend"
    exit 1
fi
