#!/bin/bash

# Virtual environment directory
VENV_DIR="venv"
ENV_FILE=".env"

activate_venv() {
    # Check if the virtual environment exists
    if [ ! -d "$VENV_DIR" ]; then
        echo "Virtual environment not found. Create it with: python -m venv $VENV_DIR"
        exit 1
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Load environment variables from .env if available
    if [ -f "$ENV_FILE" ]; then
        export $(grep -v '^#' "$ENV_FILE" | xargs)
        echo "Loaded environment variables from .env."
    else
        echo "Warning: .env file not found."
    fi
}

deactivate_venv() {
    # Unset environment variables set from .env
    if [ -f "$ENV_FILE" ]; then
        while IFS='=' read -r key _; do
            unset "$key"
        done < <(grep -v '^#' "$ENV_FILE")
        echo "Unset environment variables from .env."
    fi

    # Deactivate virtual environment
    deactivate
}

if [ "$1" == "activate" ]; then
    activate_venv
elif [ "$1" == "deactivate" ]; then
    deactivate_venv
else
    echo "Usage: $0 {activate|deactivate}"
fi
