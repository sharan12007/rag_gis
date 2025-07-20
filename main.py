from langchain.chat_models import init_chat_model 
from dotenv import load_dotenv
from VectorStore import Vector_Store_Check
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
import os
load_dotenv()

#environment variables
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

#llm initialization
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

#Urls
urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
vector_store=Vector_Store_Check(urls)
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
    content="\n\n".join([docs.page_content for docs in state["context"]])
    messages=prompt.invoke({"question":state["question"], "context":content})
    llm_answer=llm.invoke(messages)
    return {"answer":llm_answer.content}

#LangGraph definition
graph_builder=StateGraph(State).add_sequence([retriver, generate])
graph_builder.add_edge(START, "retriver")
graph=graph_builder.compile()

response = graph.invoke({"question": "What is Task Decomposition?"},stream_mode="values")
print(response["answer"])

