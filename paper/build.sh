#!/bin/bash
# Build the EzhuthuBench arXiv paper
# Run on A100 (Linux) with texlive installed

set -e
cd "$(dirname "$0")"

# Install texlive if needed
if ! command -v pdflatex &> /dev/null; then
    echo "Installing texlive..."
    apt-get update -q && apt-get install -y -q \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-science \
        texlive-publishers \
        latexmk
fi

echo "Building paper..."
latexmk -pdf -interaction=nonstopmode main.tex

echo ""
echo "Done: main.pdf"
ls -lh main.pdf
