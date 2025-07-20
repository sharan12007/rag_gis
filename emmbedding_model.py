# embedding_loader.py
from langchain_huggingface import HuggingFaceEmbeddings

_model = None
def get_embedding_model():
    global _model
    if _model is None:
        _model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return _model
