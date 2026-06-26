import os
import sys
import json
from pathlib import Path

# Add project root to Python search path to enable direct execution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Import state and tools
from src.state import AgentState
from src.tools import retrieve_policy_docs, compliance_fine_calculator, web_search_fallback

# Initialize Assets folder
os.makedirs("assets", exist_ok=True)

def get_llm():
    """
    Initialize and return the LLM based on the configuration in .env.
    Defaults to Gemini 2.5 Flash with Groq Llama 3.1 8B fallback.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "groq" and os.getenv("GROQ_API_KEY"):
        print("[LLM] Initializing Groq LLM (llama-3.1-8b-instant)...")
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0, timeout=20)
    else:
        print("[LLM] Initializing Gemini LLM (gemini-2.5-flash) with Groq fallback...")
        # Set a 20-second timeout to prevent indefinite network hangs
        gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, timeout=20)
        if os.getenv("GROQ_API_KEY"):
            groq_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, timeout=20)
            return gemini_llm.with_fallbacks([groq_llm], exceptions_to_handle=(Exception,))
        return gemini_llm

# ==========================================
# 1. NODES
# ==========================================

def retrieve_node(state: AgentState):
    """
    Retrieve document chunks from the local vector database.
    """
    print("\n--- [NODE] Retrieving Documents ---")
    query = state.get("query")
    steps = state.get("steps", [])
    
    # Use the retriever directly to get proper Document objects
    from src.vector_store import get_retriever
    try:
        retriever = get_retriever(k=8)
        docs = retriever.invoke(query)
        if not docs:
            docs = []
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        docs = []
    
    return {
        "documents": docs,
        "steps": steps + ["retrieve"]
    }

def web_search_node(state: AgentState):
    """
    Search the live web if local documents are graded irrelevant.
    """
    print("\n--- [NODE] Web Search Fallback ---")
    query = state.get("query")
    steps = state.get("steps", [])
    
    # Run web search
    search_results = web_search_fallback.invoke(query)
    
    # Store search results as a document in the state
    new_doc = Document(
        page_content=search_results, 
        metadata={"source": "Web Search (Tavily/DuckDuckGo)"}
    )
    
    current_docs = state.get("documents", [])
    return {
        "documents": current_docs + [new_doc],
        "steps": steps + ["web_search"]
    }

def calculator_node(state: AgentState):
    """
    Use the LLM to extract parameters, call the fine calculator, and generate the final answer.
    """
    print("\n--- [NODE] AI Act Fine Calculator ---")
    query = state.get("query")
    steps = state.get("steps", [])
    llm = get_llm()
    
    # Prompt the LLM to extract global turnover and violation type from the query
    system_prompt = (
        "You are an assistant that extracts parameters for an EU AI Act fine calculator.\n"
        "From the user query, extract:\n"
        "1. The global turnover of the company in Euros (as a float). If the user states '100 million', output 100000000.0.\n"
        "2. The type of violation as one of: 'prohibited', 'high_risk', or 'misleading'.\n\n"
        "Respond ONLY in valid JSON format with keys 'turnover' (float) and 'violation' (string). No other text."
    )
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    try:
        data = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
        turnover = float(data.get("turnover", 0.0))
        violation = str(data.get("violation", "high_risk"))
        
        # Invoke the calculator tool
        calculator_result = compliance_fine_calculator.invoke({
            "global_turnover": turnover,
            "violation_type": violation
        })
        
        # Ask LLM to format the calculator output nicely for the user
        formatting_prompt = (
            f"Present the following fine calculation results clearly to the user in a professional tone:\n\n"
            f"{calculator_result}"
        )
        final_answer = llm.invoke([formatting_prompt])
        generation = final_answer.content
        ai_message = final_answer
    except Exception as e:
        generation = f"Could not perform fine calculation. Error parsing inputs: {str(e)}"
        ai_message = AIMessage(content=generation)
        
    return {
        "generation": generation,
        "messages": [ai_message],
        "steps": steps + ["calculator"]
    }

def generate_node(state: AgentState):
    """
    Generate an answer using the retrieved documents and conversation history.
    """
    print("\n--- [NODE] Generating Answer ---")
    docs = state.get("documents", [])
    messages = state.get("messages", [])
    steps = state.get("steps", [])
    query = state.get("query", "")
    
    # Format the retrieved documents context
    context = ""
    if docs:
        context = "\n\n".join([f"Source: {d.metadata.get('source')}\nContent: {d.page_content}" for d in docs])
        
    system_prompt = (
        "You are an expert AI Ethics and Regulations Advisor specializing in the EU AI Act and OECD AI Principles.\n"
        "Answer the user's question using the provided document context below.\n"
        "Synthesize information from ALL provided document chunks to build a comprehensive answer.\n"
        "IMPORTANT: Your response must be strictly grounded in the provided context. Do NOT include any external knowledge, facts, dates, or examples that are not mentioned in the context. If you do, the system will flag it as a hallucination.\n\n"
        f"--- CONTEXT ---\n{context}\n----------------\n"
    )
    
    # Filter messages up to the last HumanMessage to discard any intermediate draft AIMessages
    clean_messages = []
    last_human_idx = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_human_idx = i
            
    if last_human_idx != -1:
        clean_messages = messages[:last_human_idx + 1]
    else:
        # Fallback if no human message in list
        clean_messages = [HumanMessage(content=query)]
        
    llm = get_llm()
    # We pass the conversation history messages, prepended with our system instruction
    chat_history = [SystemMessage(content=system_prompt)] + clean_messages
    response = llm.invoke(chat_history)
    
    return {
        "generation": response.content,
        "messages": [response],
        "steps": steps + ["generate"]
    }

def reformulate_node(state: AgentState):
    """
    Reformulate the query into a better search phrase if document retrieval was poor.
    """
    print("\n--- [NODE] Reformulating Query ---")
    query = state.get("query")
    steps = state.get("steps", [])
    llm = get_llm()
    
    prompt = (
        f"Analyze the user's query: '{query}'\n"
        f"Reformulate it into a precise, keyword-rich search phrase optimized for retrieving information "
        f"about the EU AI Act or OECD AI Principles from a vector database.\n"
        f"Respond ONLY with the reformulated query text."
    )
    
    response = llm.invoke(prompt)
    new_query = response.content.strip()
    print(f"Old query: '{query}' -> Reformulated: '{new_query}'")
    
    return {
        "query": new_query,
        "steps": steps + ["reformulate"]
    }

# ==========================================
# 2. CONDITIONAL EDGES / ROUTERS
# ==========================================

def route_question(state: AgentState) -> Literal["calculator", "vector_store", "direct"]:
    """
    Route the question to fine calculator, document retrieval, or direct answering.
    """
    print("--- [ROUTER] Routing Question ---")
    query = state.get("query")
    llm = get_llm()
    
    prompt = (
        "Analyze the user query and classify it into one of these three categories:\n"
        "1. 'calculator': If the query mentions calculating, computing, or estimating administrative fines or penalties "
        "under the EU AI Act (e.g. mentions of turnover, company size, percentages, or fines).\n"
        "2. 'vector_store': If the query asks factual regulatory questions about the EU AI Act, risk levels, "
        "obligations, OECD principles, or compliance requirements.\n"
        "3. 'direct': If the query is just a greeting, parting, or general conversation that requires no external policy information.\n\n"
        "Respond with ONLY one word: 'calculator', 'vector_store', or 'direct'."
    )
    
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
    decision = response.content.strip().lower().replace("'", "").replace('"', "")
    print(f"Router Decision: {decision}")
    
    if decision in ["calculator", "vector_store", "direct"]:
        return decision
    return "vector_store"  # Fallback

def grade_documents(state: AgentState) -> Literal["generate", "web_search"]:
    """
    Evaluate retrieved documents for relevance to the query.
    If irrelevant, trigger web search fallback.
    """
    print("--- [ROUTER] Grading Document Relevance ---")
    query = state.get("query")
    docs = state.get("documents", [])
    
    if not docs:
        print("No documents found. Routing to web search.")
        return "web_search"
        
    llm = get_llm()
    prompt = (
        "You are a strict evaluator grading document relevance to a user query.\n"
        "Analyze the following retrieved documents and decide if they contain specific information "
        f"that is useful or directly relevant to answering the query: '{query}'\n\n"
        "Be very strict: a document is only relevant if it contains information that can help directly answer "
        "the specific question. If the document is about the same general topic (like the EU AI Act) but contains "
        "absolutely no specific information to address the user's question (like the status of the EU AI Office in 2026), "
        "it is NOT relevant and you must grade it 'no'.\n\n"
        "Respond with 'yes' if at least one document is relevant, or 'no' if all documents are irrelevant.\n"
        "Respond with ONLY 'yes' or 'no'."
    )
    
    context = "\n\n".join([d.page_content for d in docs])
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=context)])
    grade = response.content.strip().lower()
    print(f"Document Relevance Grade: {grade}")
    
    if "yes" in grade:
        return "generate"
    return "web_search"

def grade_generation(state: AgentState) -> Literal["end", "regenerate", "reformulate", "web_search"]:
    """
    Check the generated answer for hallucinations (groundedness) and completeness.
    """
    print("--- [ROUTER] Grading Answer (Self-Correction) ---")
    query = state.get("query")
    docs = state.get("documents", [])
    generation = state.get("generation", "")
    
    if not docs:
        return "end"
        
    llm = get_llm()
    
    # 1. Hallucination Check (Is the response supported by the documents?)
    hallucination_prompt = (
        "You are an auditor grading if an AI's response contains hallucinations.\n"
        "Compare the response to the provided document context.\n"
        "The response is grounded if its core claims, definitions, and facts are supported by the context.\n"
        "Do not penalize minor elaboration, stylistic differences, or synonyms.\n"
        "Only grade 'no' if the response asserts significant factual claims, numbers, or rules that are completely absent and unsupported by it.\n"
        "Respond with ONLY 'yes' if the response is grounded (no hallucinations), or 'no' if it is hallucinated."
    )
    
    context = "\n\n".join([d.page_content for d in docs])
    response_hallucination = llm.invoke([
        SystemMessage(content=hallucination_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nResponse:\n{generation}")
    ])
    is_grounded_text = response_hallucination.content.strip().lower()
    
    # Robust parsing: check if model responds with 'yes' or positive phrases
    is_grounded = "no"
    if "yes" in is_grounded_text and "no" not in is_grounded_text.split()[:2]:
        is_grounded = "yes"
    print(f"Is answer grounded in docs? (No hallucinations): {is_grounded} (raw: {is_grounded_text})")
    
    if is_grounded == "no":
        steps = state.get("steps", [])
        if steps.count("generate") >= 3:
            print("Already regenerated answer 3 times. Ending to prevent infinite loop.")
            return "end"
        print("Hallucination detected! Routing back to regenerate...")
        return "regenerate"
        
    # 2. Answer Completeness Check (Does it answer the user's prompt?)
    answer_prompt = (
        "You are an evaluator checking if a generated response addresses the user's question.\n"
        f"The user asked: '{query}'\n"
        "Does the response provide a meaningful, substantive answer to the core intent of the query?\n"
        "A response is acceptable even if it doesn't cover every possible detail, as long as it addresses the main question.\n"
        "A response that says 'I don't have enough information' or refuses to answer should be graded 'no'.\n"
        "Respond with ONLY 'yes' or 'no'."
    )
    response_completeness = llm.invoke([
        SystemMessage(content=answer_prompt),
        HumanMessage(content=f"Response:\n{generation}")
    ])
    answers_query_text = response_completeness.content.strip().lower()
    
    # Robust parsing
    answers_query = "no"
    if "yes" in answers_query_text and "no" not in answers_query_text.split()[:2]:
        answers_query = "yes"
    print(f"Does answer address user query?: {answers_query} (raw: {answers_query_text})")
    
    if answers_query == "no":
        steps = state.get("steps", [])
        # If web search hasn't been tried yet, route to web search as fallback
        if "web_search" not in steps:
            print("Answer is incomplete and web search not yet tried. Routing to web search fallback...")
            return "web_search"
        # If we have already reformulated once, let's stop to prevent infinite loops
        if steps.count("reformulate") >= 1:
            print("Already reformulated query. Ending to avoid infinite loop.")
            return "end"
        print("Answer is incomplete. Routing to query reformulation...")
        return "reformulate"
        
    print("Answer passed all grades successfully!")
    return "end"

# ==========================================
# 3. BUILD THE GRAPH
# ==========================================

def compile_graph():
    """
    Construct, compile, and return the LangGraph Agentic RAG workflow.
    """
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("calculator", calculator_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("reformulate", reformulate_node)
    
    # Define Entry and Conditional Edges
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "calculator": "calculator",
            "vector_store": "retrieve",
            "direct": "generate"
        }
    )
    
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate": "generate",
            "web_search": "web_search"
        }
    )
    
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("calculator", END)
    workflow.add_edge("reformulate", "retrieve")
    
    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "end": END,
            "regenerate": "generate",
            "reformulate": "reformulate",
            "web_search": "web_search"
        }
    )
    
    # Configure conversation checkpoint memory
    memory = MemorySaver()
    compiled_workflow = workflow.compile(checkpointer=memory)
    
    # Save graph diagram to assets folder
    try:
        graph_img = compiled_workflow.get_graph().draw_mermaid_png()
        with open("assets/graph_architecture.png", "wb") as f:
            f.write(graph_img)
        print("Visual graph architecture saved to assets/graph_architecture.png")
    except Exception as e:
        print(f"[Warning] Could not generate graph PNG: {e}. You may need to install pygraphviz.")
        
    return compiled_workflow

if __name__ == "__main__":
    # Test script: allows testing compilation and running queries from command line
    print("Compiling LangGraph workflow...")
    agent = compile_graph()
    
    # Test a compliance question
    test_query = "What is the penalty if a company with a global turnover of 100M violates prohibited AI rules?"
    print(f"\n--- Running Graph Test with query: '{test_query}' ---")
    
    config = {"configurable": {"thread_id": "test_thread"}}
    initial_state = {
        "query": test_query,
        "messages": [HumanMessage(content=test_query)],
        "documents": [],
        "generation": "",
        "steps": []
    }
    
    events = agent.stream(initial_state, config=config)
    for event in events:
        print(event)
