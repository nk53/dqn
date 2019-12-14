#!/bin/bash

if [ "$CONDA_DEFAULT_ENV" != "kgpu" ]; then
    echo "The kgpu environment is not active."
    echo "First activate kgpu environment, then run $0"
    exit 1
fi

pip install tf-nightly-gpu tf-nightly keras
