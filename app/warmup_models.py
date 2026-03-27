from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

print("Downloading embedding model...")
HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"}
)
print("Downloading reranker model...")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("All models cached.")
