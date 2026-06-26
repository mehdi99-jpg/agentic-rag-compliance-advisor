import os
import sys
from pathlib import Path

# Add project root to Python search path to enable direct execution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

# We import the retriever from our vector store module
from src.vector_store import get_retriever

@tool
def retrieve_policy_docs(query: str) -> str:
    """
    Search the local policy database for information regarding AI regulations, 
    risk classifications, compliance requirements, and penalties under the EU AI Act 
    and OECD AI Principles. Use this tool as the primary source of knowledge.
    """
    try:
        retriever = get_retriever(k=3)
        docs = retriever.invoke(query)
        if not docs:
            return "No matching documents found in the local policy database."
            
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            formatted_docs.append(f"[Document Chunk {i+1}] (Source: {source})\nContent:\n{doc.page_content}\n")
        return "\n---\n".join(formatted_docs)
    except Exception as e:
        return f"Error retrieving documents from local database: {str(e)}"

@tool
def compliance_fine_calculator(global_turnover: float, violation_type: str) -> str:
    """
    Calculate the maximum potential administrative fine under the EU AI Act.
    
    Arguments:
    - global_turnover: The total global annual turnover of the company in Euros (e.g. 100000000.0).
    - violation_type: The type of violation. Must be one of the following strings:
      * 'prohibited' (for unacceptable practices like social scoring or cognitive manipulation)
      * 'high_risk' (for failing to meet obligations for high-risk systems, transparency rules, etc.)
      * 'misleading' (for supplying incorrect, incomplete, or misleading information to authorities)
    """
    # Standardize the input violation type
    v_type = violation_type.lower().strip()
    
    # Define thresholds
    if v_type in ["prohibited", "unacceptable", "banned"]:
        percentage = 0.07  # 7%
        fixed_fine = 35000000.0  # €35 Million
        desc = "Non-compliance with Unacceptable/Prohibited AI Practices (e.g., social scoring, cognitive manipulation)"
    elif v_type in ["high_risk", "highrisk", "obligations", "transparency"]:
        percentage = 0.03  # 3%
        fixed_fine = 15000000.0  # €15 Million
        desc = "Non-compliance with High-Risk AI system requirements or Transparency rules"
    elif v_type in ["misleading", "incorrect", "incomplete", "information"]:
        percentage = 0.015  # 1.5%
        fixed_fine = 7500000.0  # €7.5 Million
        desc = "Supplying incorrect, incomplete, or misleading information to regulators"
    else:
        return (
            "Error: Invalid violation_type. Please select one of the following: "
            "'prohibited', 'high_risk', or 'misleading'."
        )
        
    # Calculation: whichever is HIGHER
    turnover_fine = global_turnover * percentage
    max_fine = max(turnover_fine, fixed_fine)
    
    return (
        f"--- EU AI Act Fine Calculation Result ---\n"
        f"Violation Type: {desc}\n"
        f"Company Annual Global Turnover: €{global_turnover:,.2f}\n"
        f"Fine Rule: {percentage*100}% of turnover or €{fixed_fine:,.2f} (whichever is HIGHER)\n"
        f"Calculated Turnover-based Fine: €{turnover_fine:,.2f}\n"
        f"Fixed Fine Threshold: €{fixed_fine:,.2f}\n"
        f"RESULT (Maximum Fine Applicable): €{max_fine:,.2f}"
    )

@tool
def web_search_fallback(query: str) -> str:
    """
    Search the live web for general information, recent news, or details 
    not found in the local policy documents. Use this tool only when the 
    local database retrieval does not yield relevant answers.
    """
    # Check if Tavily API key is set
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if tavily_key and tavily_key != "YOUR_TAVILY_API_KEY_HERE":
        try:
            print(f"[TOOL] Running web search via Tavily: '{query}'")
            search = TavilySearchResults(max_results=3)
            results = search.invoke(query)
            
            # Format Tavily results
            formatted_results = []
            for i, res in enumerate(results):
                formatted_results.append(
                    f"[Web Search Result {i+1}]\nURL: {res.get('url')}\nContent: {res.get('content')}\n"
                )
            return "\n---\n".join(formatted_results)
        except Exception as e:
            print(f"[TOOL] Tavily search failed: {e}. Falling back to DuckDuckGo...")
            
    # Fallback to DuckDuckGo search (no API key required, completely free)
    try:
        print(f"[TOOL] Running web search via DuckDuckGo: '{query}'")
        search = DuckDuckGoSearchRun()
        return search.invoke(query)
    except Exception as e:
        return f"Error executing web search: {str(e)}"

if __name__ == "__main__":
    # Test script: allows verifying that the tools run correctly in isolation
    print("Testing tools...")
    
    # 1. Test calculation tool
    test_turnover = 100000000.0  # €100 Million
    print("\n--- Testing calculator tool ---")
    print(compliance_fine_calculator.invoke({"global_turnover": test_turnover, "violation_type": "prohibited"}))
    
    # 2. Test search tool
    print("\n--- Testing search tool ---")
    print(web_search_fallback.invoke({"query": "What is the latest status of the EU AI Office in 2026?"}))
