#!/bin/bash
echo "Building project packages"
python3 -m pip install -r requirements.txt

echo "Building project packages..."
python3 -m pipenv install --system --deploy --ignore-pipfile

echo "Creating dist folder..."
mkdir -p dist
touch dist/dummy_file.py

echo "Setup complete"
