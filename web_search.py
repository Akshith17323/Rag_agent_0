import os
import requests

def search_web(query: str) -> list[dict]:
    """
    Perform a web search using the Serper API.
    
    Args:
        query (str): The search term.
        
    Returns:
        list[dict]: A list of up to 5 result dictionaries containing 'title', 'snippet', and 'link'.
                    Returns an empty list on any error.
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        print("Error: SERPER_API_KEY not found in environment.")
        return []

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": 5
    }

    try:
        # 10-second timeout for the request
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract the organic search results, defaulting to empty list if not found
        organic_results = data.get("organic", [])
        
        # Format the results into a list of dictionaries with specific keys
        formatted_results = []
        for item in organic_results[:5]: # Ensure we only take up to 5 results
            formatted_results.append({
                "title": item.get("title", "No Title"),
                "snippet": item.get("snippet", "No Snippet"),
                "link": item.get("link", "#")
            })
            
        return formatted_results

    except requests.exceptions.Timeout:
        print("Error: Search request timed out.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Error during search request: {e}")
        return []
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         return []

def format_search_results(results: list) -> str:
    """
    Format a list of search result dictionaries into a clean string.
    
    Args:
        results (list): List of dictionaries containing search results.
        
    Returns:
        str: A formatted string of results, or a fallback message if empty.
    """
    if not results:
        return "No web results found."

    formatted_str = ""
    for i, result in enumerate(results, start=1):
        formatted_str += f"[{i}] Title: {result.get('title')}\n"
        formatted_str += f"    Summary: {result.get('snippet')}\n"
        formatted_str += f"    Source: {result.get('link')}\n\n"
        
    return formatted_str.strip()

if __name__ == "__main__":
    # Test block to verify functionality
    test_query = "What is LangChain?"
    print(f"Searching for: '{test_query}'...\n")
    
    # Perform the search
    search_results = search_web(test_query)
    
    # Format and print the results
    formatted_output = format_search_results(search_results)
    
    print(formatted_output)
