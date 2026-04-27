import wikipedia
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WikipediaSearch") 

async def list_prompts(session):
    prompt_response = await session.list_prompts()

    if not prompt_response or not prompt_response.prompts:
        print("No prompts found on the MCP server.")
        return
    
    print("Available prompts on the MCP server:")
    for p in prompt_response.prompts:
        print(f"\nPrompt Name: {p.name}")
        if p.arguments:
            for arg in p.arguments:
                print(f"  - {arg.name}")
        else:
            print("  (No arguments)")
        print("\nUse: /prompt <prompt_name> \"arg1\" \"arg2\" ...")    

@mcp.prompt()

def wikipedia_search_prompt(query: str) -> str:
    """
    Prompt template for searching Wikipedia.
    """
    return f"""
    Search Wikipedia for: {query}". 
    Provide a concise summary of the topic, including the title and a URL to the page.
    If the topic is ambiguous, suggest possible alternatives.
    """

@mcp.tool()

def fetch_wikipedia_info(query: str) -> dict:
    """
    Search Wikipedia for a topic and return title, summary, and URL of the best match.
    """
    try:
        search_results = wikipedia.search(query, results=5)
        if not search_results:
            return {"error": "No results found for your query."}

        best_match = search_results[0]
        page = wikipedia.page(best_match)

        return {
            "title": page.title,
            "summary": page.summary,
            "url": page.url
        }

    except wikipedia.DisambiguationError as e:
        return {
            "error": f"Ambiguous topic. Try one of these: {', '.join(e.options[:5])}"
        }

    except wikipedia.PageError:
        return {
            "error": "No Wikipedia page could be loaded for this query."
        }


# Run the MCP server
if __name__ == "__main__":
    print("Starting MCP Wikipedia Server...")
    mcp.run(transport="stdio")
    