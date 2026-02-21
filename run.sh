#!/bin/bash

# Farm Assistant Run Script
# Usage: ./run.sh [local|docker]

MODE=${1:-local}

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
    
    # Run the application
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --timeout-keep-alive 120
else
    echo "Usage: ./run.sh [local|docker]"
    echo ""
    echo "Options:"
    echo "  local   - Run with Uvicorn (development)"
    echo "  docker  - Run with Docker Compose"
    exit 1
fi
