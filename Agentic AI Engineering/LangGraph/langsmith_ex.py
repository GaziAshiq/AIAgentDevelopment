from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
load_dotenv(override=True)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model="openai:gpt-5-mini",
    tools=[get_weather],
    prompt="You are a helpful assistant.",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
)