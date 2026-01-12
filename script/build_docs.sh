#!/bin/bash
set -e

# Build documentation script for Diffulex
# This script can be used both locally and in CI environments

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Check if we're in a CI environment (GitHub Actions provides this)
if [ -z "${CI}" ] && [ -z "${GITHUB_ACTIONS}" ]; then
    # Local development: create and use virtual environment
    if [ ! -d ".venv" ]; then
        python -m venv .venv
    fi
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r docs/requirements.txt
else
    # CI environment: just install dependencies (Python environment already set up)
    pip install --upgrade pip
    pip install -r docs/requirements.txt
fi

# Build documentation
cd docs
make html

# Create .nojekyll file to disable Jekyll processing on GitHub Pages
touch _build/html/.nojekyll
echo ".nojekyll file created to disable Jekyll"

# Copy CNAME file if it exists (for GitHub Pages custom domain)
if [ -f "CNAME" ]; then
    cp CNAME _build/html/
    echo "CNAME file copied to build output"
fi

echo "Documentation build completed successfully!"

