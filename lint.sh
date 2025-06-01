#!/bin/bash
# Run Pylint on all Python files in the repo

pip install -q pylint
pylint $(git ls-files '*.py')
