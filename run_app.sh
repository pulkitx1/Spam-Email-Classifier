#!/bin/bash
# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate venv and install requirements
source venv/bin/activate
pip install -r requirements.txt

# Run the app
python3 app.py
