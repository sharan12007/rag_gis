from urlAvailableCheck import url_valid_check
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
from emmbedding_model import get_embedding_model

def Vector_Store_Check(urls: list) -> Chroma:
    # Initialize embeddings and vector store
    vector_store = Chroma(
        collection_name="rag_gis",
        embedding_function=get_embedding_model(),
        persist_directory="./chroma_db"
    )

    # Process each URL
    for url in urls:
        if url_valid_check(url):  # Only proceed if new or changed
            print(f"Processing URL: {url}")
            loader = WebBaseLoader(
                web_path=(url,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
            docs = loader.load()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(docs)

            # Add to vector store
            _ = vector_store.add_documents(docs)

    # Persist after all URLs are processed
    
    return vector_store
