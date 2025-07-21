from langchain.chat_models import init_chat_model 
from dotenv import load_dotenv
from VectorStore import Vector_Store_Check
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph,MessagesState
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode,tools_condition
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


#LangGraph definition
class State(TypedDict):
    question: str
    answer: str
    context: List[Document]
    
#Functions for retriver and generator
@tool(response_format="content_and_artifact")
def retriver(query : str):
    """Retrieves relevant context for the query."""
    retrieved_docs=vector_store.similarity_search(query, k=3)
    serialized_docs = "\n\n".join((f"Source : {doc.metadata}\n Content : {doc.page_content}") for doc in retrieved_docs)
    return serialized_docs , retrieved_docs

#chatbot node
def chatbot(state: MessagesState):
    """" Generate a tool call or respond"""
    llm_with_tools=llm.bind_tools([retriver],
        system_message=SystemMessage(content=(
        "You are an AI assistant. For every user query, if possible, you MUST first call the 'retriver' tool to fetch relevant information, "
        "even if you think you know the answer. Only skip tool calls if absolutely sure."
    )),tool_choice="auto")
    response=llm_with_tools.invoke(state["messages"])
    return {"messages":[response]}
#tool node
tools=ToolNode([retriver])

#generator function
def generate(state:MessagesState):
    """Generate answer."""
    recent_tool_calls=[]
    for message in reversed(state["messages"]):
        if message.type == "tool" :
            recent_tool_calls.extend(message)
        else:
            break
    tool_messages=recent_tool_calls[::-1]

    docs_content="\n\n".join(content if isinstance(content, str) else str(content)
    for content, _ in tool_messages)
    systemmessagecontent=(
         "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
    message
    for message in state["messages"]
    if message.type in ("human", "system")
    or (message.type == "ai" and not message.tool_calls)
]
    prompts=[SystemMessage(content=systemmessagecontent)]+ conversation_messages
    response=llm.invoke(prompts)
    return {"messages": [response]}


#LangGraph definition
graph_builder=StateGraph(MessagesState)
graph_builder.add_node(chatbot)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.set_entry_point("chatbot")
graph_builder.add_conditional_edges(
    "chatbot",tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools","generate")
graph_builder.add_edge("chatbot", END)

graph=graph_builder.compile()

chk=True
print("Welcome to the RAG GIS Chatbot! Type 'exit' to quit.")
while chk:
    try:
        user_input= input("user: ")
        if(user_input.lower() == "exit"):
            chk=False
            print("Exiting the chatbot. Goodbye!")
            break
        for steps in graph.stream(
            {"messages":[{"role":"user","content":user_input}]},stream_mode="values"
        ):
            for message in steps["messages"]:
                if message.type == "ai":
                    print(f"AI: {message.content}")
                
    except Exception as e:
        print(f"An error occurred: {e}")


