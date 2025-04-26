#!/bin/bash
# CREATE A LOCAL ENV
MYVENV="venv"
python3 -m venv $MYVENV
source $MYVENV/bin/activate
pip install -r requirements.txt
