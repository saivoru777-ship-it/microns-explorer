#!/bin/bash
# Build paper PDF. Usage: ./build.sh
cd "$(dirname "$0")" && tectonic main.tex 2>&1 | grep -v "internal consistency" | grep -v "Rerunning TeX because"
