from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from VectorStore import Vector_Store_Check
from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch
import os

load_dotenv()

# Environment variables
os.environ["USER_AGENT"] = "RAG-GIS-Agent/1.0 (Linux; Python)"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# LLM initialization
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Setup vector store from URLs
urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
vector_store = Vector_Store_Check(urls)

@tool
def search_web(query: str) -> str:
    """Search the web for real-time or latest info on anything (news, weather, movies, etc)."""
    search = TavilySearch(max_results=3)
    search_results = search.invoke(query)
    return search_results

@tool
def retriver(query: str):
    """Retrieves relevant context for the query from internal documents."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized_docs = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
    )
    return serialized_docs



def chatbot(state: MessagesState):
    """Initial node: Triggers a tool call or generates a direct response."""
    llm_with_tools = llm.bind_tools(
        [retriver, search_web],
        system_message=SystemMessage(content=(
            """
           Answer the following questions as best you can. You have access to the following tools:

              1. **retriver**: Use this tool to retrieve relevant context from internal documents.  
                2. **search_web**: Use this tool to search the web for real-time or latest info on anything (news, weather, movies, etc).
              If you need to use a tool, respond with a message that includes the tool call.
"""
        )),
        tool_choice="auto"
    )
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Tool node
tools = ToolNode([retriver, search_web])

def generate(state: MessagesState):
    results = state["messages"][-1].content
    """Generates a response based on the last message content."""
    return{"messages": [results]}

# Memory & config
memory = MemorySaver()
config = {"configurable": {"thread_id": "abc123"}}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", tools)
builder.add_node("generate", generate)
builder.set_entry_point("chatbot")
builder.add_conditional_edges("chatbot", tools_condition, {END: END, "tools": "tools"})
builder.add_edge("tools", "generate")
builder.add_edge("generate","chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile(checkpointer=memory)
graph.get_graph().print_ascii()
print("Welcome to the RAG GIS Chatbot with Web Search! Type 'exit' to quit.")
while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Exiting the chatbot. Goodbye!")
            break
        
        last_ai_msg = None
        used_tools = set()

        for step in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config=config,
        ):
            for msg in step["messages"]:
                if msg.type == "tool":
                    used_tools.add(msg.name)
                elif msg.__class__.__name__ == "AIMessage" and not getattr(msg, "tool_calls", None):
                    last_ai_msg = msg.content.strip()

        if used_tools:
            print("\n🔧 Tools used:", ", ".join(sorted(used_tools)))
        if last_ai_msg:
            print("\n🤖 AI:", last_ai_msg)
        else:
            print("\n🤖 AI: [No direct response — tool was used or no final message]")

