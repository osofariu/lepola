#!/bin/zsh

# analyze document

function analyze_document() {
    local document_id=$1
    curl -X 'POST' \
        "http://localhost:8000/api/v1/pipeline/analyze/${document_id}" \
    -H 'accept: application/json' \
    -d ''
}

function main() {
    analyze_document "f88cb1d4-30b5-4888-bfca-8f5bec74af77"
}


