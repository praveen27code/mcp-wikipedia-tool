import asyncio
import os
import shlex
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from langchain_mcp_adapters.tools import load_mcp_tools

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MCP server launch config
server_params = StdioServerParameters(
    command="python",
    args=["mcp-server.py"]
)

# LangGraph state definition
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


async def create_graph(session):
    # Load tools from MCP server
    tools = await load_mcp_tools(session)

    # LLM configuration (get API key from environment variable)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)

    # Prompt template with user/assistant chat only
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that uses tools to search Wikipedia."),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    # Define chat node
    def chat_node(state: State) -> State:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state

    # Build LangGraph with tool routing
    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())

async def handle_resource(session, command):
    parts = shlex.split(command)
    if len(parts) < 2:
        print("Usage: /resource <name>")
        return
    resource_id = parts[1]
    try:
        # Get all available resources
        response = await session.list_resources()
        resources = response.resources
        resource_map = {str(i + 1): r.name for i, r in enumerate(resources)}

        # Resolve name or index
        resource_name = resource_map.get(resource_id, resource_id)
        match = next((r for r in resources if r.name == resource_name), None)

        if not match:
            print(f"Resource '{resource_id}' not found.")
            return

        # Fetch resource content
        result = await session.read_resource(match.uri)

        for content in result.contents:
            if hasattr(content, "text"):
                print("\n=== Resource Text ===")
                print(content.text)


    except Exception as e:
        print("Error fetching resource:", str(e))

async def list_resource(session):
    try:
        resource = await session.list_resources()
        if not resource or not resource.resources:
            print("No resources found on the MCP server.")
            return
        print("Available resources on the MCP server:")
        for i, r in enumerate(resource.resources, 1):
            print(f"[{i}] {r.name}")
        print("\nUse: /resource <name> to view its content.")
    except Exception as e:
        print("Error fetching resources:", str(e))


# Entry point
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            agent = await create_graph(session)
            print("Wikipedia MCP agent is ready.")

            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    break
                elif user_input.startswith("/resources"):
                    parts = user_input.split()
                    if len(parts) == 1:
                        # No arguments: list all resources
                        await list_resource(session)
                    else:
                        # With arguments: fetch specific resource
                        await handle_resource(session, user_input)
                    continue
                if not user_input:
                    continue

                try:
                    response = await agent.ainvoke(
                        {"messages": [HumanMessage(content=user_input)]},
                        config={"configurable": {"thread_id": "wiki-session"}}
                    )
                    # Extract the last message from the response
                    last_message = response["messages"][-1]
                    if hasattr(last_message, 'content'):
                        print("AI:", last_message.content)
                    else:
                        print("AI:", str(last_message))
                except Exception as e:
                    print("Error:", str(e))


if __name__ == "__main__":
    asyncio.run(main())