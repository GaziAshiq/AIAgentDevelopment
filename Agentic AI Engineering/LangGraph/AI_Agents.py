import os
from dotenv import load_dotenv
from typing import TypedDict, Literal, Union, Annotated, Optional, Sequence

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

load_dotenv(override=True)


def get_client(
    client_name: Literal["openai", "deepseek", "gemini"],
) -> ChatOpenAI | None:
    """
    Get a ChatOpenAI client for the specified client name.
    :param client_name: The name of the client to get.
    :return: The ChatOpenAI client.
    """
    # ====> For OpenAI models <====#
    if client_name == "openai":
        openai_client = ChatOpenAI(model="gpt-4.1-mini")
        return openai_client

    # ====> For DeepSeek models <====#
    elif client_name == "deepseek":
        deepseek_client = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/",
            temperature=0.7,
            max_tokens=512,
            timeout=None,
            max_retries=2,
        )
        return deepseek_client

    # ====> For Google Gemini models <====#
    elif client_name == "gemini":
        gemini_client = ChatOpenAI(
            model="gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=0.7,
            max_tokens=512,
            timeout=None,
            max_retries=2,
        )
        return gemini_client
    else:
        print(f"Unsupported client: {client_name}")
        return None


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def addition(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def subtraction(a: int, b: int) -> int:
    """Subtract the second number from the first number."""
    return a - b


@tool
def multiplication(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@tool
def division(a: float, b: float) -> float:
    """Divide the first number by the second number."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@tool
def get_current_datetime() -> str:
    """Get the current date and time in a readable format."""
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def search_web(query: str) -> str:
    """Search the web for information about a query."""
    # This is a mock implementation - in a real scenario you'd use an actual search API
    return f"Mock web search results for '{query}': This is a placeholder. In a real implementation, you would integrate with a search API like Google Search, Bing, or DuckDuckGo."


tools = [
    addition,
    subtraction,
    multiplication,
    division,
    get_current_datetime,
    search_web,
]
tool_node = ToolNode(tools)

model = get_client("deepseek").bind_tools(tools=tools)


def call_model(state: AgentState) -> AgentState:
    """Node that calls the model with appropriate system prompt and handles responses."""
    system_prompt = SystemMessage(
        content="""You are a helpful AI assistant with access to several tools. Use the tools available to you to answer questions accurately.

Available tools:
- Mathematical operations: addition, subtraction, multiplication, division
- Time: get_current_datetime (for current date and time)
- Information: search_web (for searching the web)

When you need to use a tool, make a tool call. If you can answer directly without tools, provide a direct response.

For calculations, use the appropriate mathematical tools. For questions about current time, use get_current_datetime. For general knowledge questions, you can search the web.

Always respond in English unless specifically asked otherwise."""
    )

    messages = [system_prompt] + state["messages"]
    response = model.invoke(messages)

    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", END]:
    """Decide whether to continue reasoning or end the conversation."""
    last_message = state["messages"][-1]

    # If the last message has tool calls, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, end the conversation
    return END


# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("call_model", call_model)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "call_model")
graph.add_conditional_edges("call_model", should_continue)
graph.add_edge("tools", "call_model")  # After tool execution, go back to model

app = graph.compile()

if __name__ == "__main__":
    conversations_history = []

    print("🤖 AI Assistant (with Tools)")
    print("=========================")
    print("Ask me anything! I can use tools to help answer your questions.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break

        # Add user message to conversation history
        conversations_history.append(HumanMessage(content=user_input))

        # Get AI response using LangGraph
        result = app.invoke({"messages": conversations_history})

        # Extract the latest AI response
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            latest_ai_message = ai_messages[-1].content
            print(f"🤖 AI: {latest_ai_message}")

            # Check if the AI made any tool calls
            if hasattr(ai_messages[-1], "tool_calls") and ai_messages[-1].tool_calls:
                print(
                    f"🔧 Used tools: {[call['name'] for call in ai_messages[-1].tool_calls]}"
                )

        # Update conversation history with all messages from result
        conversations_history = result["messages"]

        print()  # Add spacing between turns
