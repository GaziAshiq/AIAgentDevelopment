from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    num1: int
    operation: str
    num2: int
    result: int

def adder(state: AgentState) -> AgentState:
    """sum of num1 and num2"""
    state['result'] = state['num1'] + state['num2']
    return state

def subtractor(state: AgentState) -> AgentState:
    """subtraction of num1 and num2"""
    state['result'] = state['num1'] - state['num2']
    return state

def router_node(state: AgentState) -> AgentState:
    """Router node that passes state through for conditional routing"""
    return state

def decide_next_node(state: AgentState) -> str:
    """this node will select the next node of the graph"""
    if state['operation'] == "+":
        return "adder_node"
    elif state['operation'] == "-":
        return "subtractor_node"
    else:
        return "adder_node"  # default fallback

graph = StateGraph(AgentState)

graph.add_node("router_node", router_node)
graph.add_node("adder_node", adder)
graph.add_node("subtractor_node", subtractor)

graph.add_edge(START, "router_node")
graph.add_conditional_edges("router_node", decide_next_node,
                            {"adder_node": "adder_node", "subtractor_node": "subtractor_node"})
graph.add_edge("adder_node", END)
graph.add_edge("subtractor_node", END)

agent_dev = graph.compile()

# Example usage for debugging
if __name__ == "__main__":
    # Test the graph with addition
    result_add = agent_dev.invoke({"num1": 10, "operation": "+", "num2": 5, "result": 0})
    print("Addition result:", result_add)

    # Test the graph with subtraction
    result_sub = agent_dev.invoke({"num1": 10, "operation": "-", "num2": 5, "result": 0})
    print("Subtraction result:", result_sub)