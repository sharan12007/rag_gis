from langchain.chat_models import init_chat_model 
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import bs4
from langchain import hub
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

#llm initialization
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
#embedding model initialization
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#vector store initialization
vector_store = Chroma(
    collection_name="rag_gis",
    embedding_function=embeddings,
    persist_directory="./chroma_db")

#loaders
loader=WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title","post-header")
        )
    ),
)
#load documents
docs=loader.load()

#split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(docs)
#add documents to vector store
_= vector_store.add_documents(docs)

#rag_prompts
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.smith.langchain.com")


#LangGraph definition

class State(TypedDict):
    question: str
    answer: str
    context: List[Document]

#Functions for retriver and generator
def retriver(state:State):
    retrieved_docs=vector_store.similarity_search(state["question"], k=3)
    return {"context":retrieved_docs}
def generate(state:State):
    content="/n/n".join([docs.page_content for docs in state["context"]])
    messages=prompt.invoke({"question":state["question"], "context":content})
    llm_answer=llm.invoke(messages)
    return {"answer":llm_answer.content}

#LangGraph definition
graph_builder=StateGraph(State).add_sequence([retriver, generate])
graph_builder.add_edge(START, "retriver")
graph=graph_builder.compile()

response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])

