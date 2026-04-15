# imporot libraries
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
import sqlite3
import os

# Grop api key
GROQ_API_KEY = "Enter your api key"

# LLM 
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    long_term_memory: str


# Retriver pipline
def get_retriever():
    db = Chroma(
        persist_directory="vector_store",
        embedding_function=HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )
    )

    return db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k":5, "score_threshold":0.3}
    )

retriever = get_retriever()

# Extract long term memory from user queries
def extract_memory_node(state: State):
    user_msg = state["messages"][-1]

    prompt = [
        SystemMessage(content="Extract long-term user info. Return bullet points or EXACTLY 'NONE'."),
        user_msg
    ]

    memory = llm.invoke(prompt).content.strip()

    if memory == "NONE":
        return {}

    updated = state.get("long_term_memory", "") + "\n" + memory
    return {"long_term_memory": updated}

# Summarize the past conversation 
def summarize_node(state: State):
    msgs = state["messages"]

    if len(msgs) < 6:
        return {}

    summary = llm.invoke(
        [SystemMessage(content="Summarize conversation briefly")] + msgs[:-1]
    )

    return {"messages": [summary, msgs[-1]]}

# Rag pipline
def rag_node(state: State):
    query = state["messages"][-1].content
    docs = retriever.invoke(query)

    if not docs:
        return {}

    context = "\n".join([d.page_content for d in docs])

    return {
        "messages": [
            SystemMessage(content=f"Use this context:\n{context}")
        ]
    }


# Chat node
def chat_node(state: State):
    memory = state.get("long_term_memory", "")

    system = SystemMessage(
        content=f"You are helpful assistant.\nUser memory:\n{memory}"
    )

    response = llm.invoke([system] + state["messages"])

    return {"messages": [response]}


graph = StateGraph(State)

graph.add_node("memory", extract_memory_node)
graph.add_node("summary", summarize_node)
graph.add_node("rag", rag_node)
graph.add_node("chat", chat_node)

graph.add_edge(START, "memory")
graph.add_edge("memory", "summary")
graph.add_edge("summary", "rag")
graph.add_edge("rag", "chat")
graph.add_edge("chat", END)


conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

chatbot = graph.compile(checkpointer=memory)


if __name__ == "__main__":

    config = {"configurable": {"thread_id": "user_1"}}

    print("Chatbot ready (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        result = chatbot.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "long_term_memory": ""
            },
            config=config
        )

        print("AI:", result["messages"][-1].content)
