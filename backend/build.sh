#!/bin/bash
set -e
pip install -r requirements.txt
echo "Pre-downloading embedding model..."
python -c "
from langchain_huggingface import HuggingFaceEmbeddings
HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print('Done.')
"
