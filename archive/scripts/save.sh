#!/usr/bin/env bash

echo '# Output from the notebook (`explore.py`)' > OUTPUT.md



{
    echo 
    echo 
    echo '```text'
    python explore.py
    echo 
    echo '```'
} >> OUTPUT.md