import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python search path to enable direct execution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# 10 Simple Questions (Direct factual retrieval)
SIMPLE_QUESTIONS = [
    "What are the four risk categories defined by the EU AI Act?",
    "What is the OECD AI Principle 1.3 about?",
    "What are prohibited AI practices under the EU AI Act?",
    "Name three high-risk AI system categories under the EU AI Act.",
    "What are the transparency obligations for limited-risk AI systems?",
    "What is the OECD's policy recommendation regarding digital ecosystems?",
    "What is the definition of a minimal risk AI system under the EU AI Act?",
    "Who is responsible for complying with high-risk system requirements under the AI Act?",
    "What is the core focus of OECD Principle 1.5?",
    "What are the record-keeping and logging obligations for high-risk AI systems?"
]

# 10 Complex Questions (Multi-hop, reasoning, web-search, or calculation)
COMPLEX_QUESTIONS = [
    "If a credit scoring AI is deployed, is it high-risk? What are the obligations, and what is the fine for a €20M company violating high-risk requirements?",
    "Compare how biometric categorization based on political beliefs is classified under the EU AI Act vs. OECD principles, and calculate the maximum fine if a company with a €200M turnover violates it.",
    "What is the penalty if a company with a €40M turnover supplies misleading information to regulatory authorities under the EU AI Act?",
    "Compare the compliance requirements for a deepfake generation tool vs. an employee recruitment CV sorter under the EU AI Act.",
    "Is a chatbot for customer service considered high-risk or limited risk? Explain its transparency rules.",
    "How do the OECD's guidelines on accountability compare to the EU AI Act's human oversight rules?",
    "What is the potential fine if a company with a €300M turnover uses subliminal manipulation leading to physical harm?",
    "If a company with €10M turnover fails to implement risk management for an admissions grading AI, what is the maximum fine?",
    "What is the latest status of the EU AI Office in 2026?",  # Requires Web Search Fallback
    "Explain the fine structure under the EU AI Act and calculate the maximum fine for a company with €500M turnover under all three violation types."
]

def run_evaluation_suite():
    """
    Execute 10 simple and 10 complex questions through the compiled graph,
    collect latency and execution path metrics, and output files and charts.
    """
    # Import compiling function from graph module
    from src.graph import compile_graph
    
    agent = compile_graph()
    results = []
    
    # Process simple questions
    print("\n=== Running Evaluation: 10 Simple Questions ===")
    for idx, q in enumerate(SIMPLE_QUESTIONS):
        print(f"Testing Simple Question {idx+1}/{len(SIMPLE_QUESTIONS)}...")
        record = run_single_test(agent, q, "Simple", idx+1)
        results.append(record)
        time.sleep(3)  # Add delay to respect API rate limits (RPM)
        
    # Process complex questions
    print("\n=== Running Evaluation: 10 Complex Questions ===")
    for idx, q in enumerate(COMPLEX_QUESTIONS):
        print(f"Testing Complex Question {idx+1}/{len(COMPLEX_QUESTIONS)}...")
        record = run_single_test(agent, q, "Complex", idx+1)
        results.append(record)
        time.sleep(3)  # Add delay to respect API rate limits (RPM)
        
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV raw data
    csv_path = ASSETS_DIR / "evaluation_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nRaw metrics saved to {csv_path}")
    
    # Generate and save charts
    plot_path = generate_evaluation_plots(df)
    
    return df, plot_path

def run_single_test(agent, question: str, q_type: str, num: int):
    """
    Run a single query through the agent and record its performance.
    """
    # Clear memory checkpoints by using a unique thread ID per question
    config = {"configurable": {"thread_id": f"eval_{q_type.lower()}_{num}"}}
    
    initial_state = {
        "query": question,
        "messages": [HumanMessage(content=question)],
        "documents": [],
        "generation": "",
        "steps": []
    }
    
    start_time = time.time()
    try:
        output_state = agent.invoke(initial_state, config=config)
        latency = time.time() - start_time
        
        steps = output_state.get("steps", [])
        docs = output_state.get("documents", [])
        generation = output_state.get("generation", "")
        
        # Calculate a simple document relevance score based on whether docs were found
        relevance = 0.0
        if docs:
            # Simple keyword search relevance metric
            matched_keywords = 0
            keywords = question.lower().split()
            for doc in docs:
                content = doc.page_content.lower()
                for keyword in keywords:
                    if len(keyword) > 4 and keyword in content:
                        matched_keywords += 1
            relevance = min(1.0, matched_keywords / max(1, len(docs) * 2))
            
            # Ensure we give at least a baseline score if documents are loaded
            if relevance == 0:
                relevance = 0.5
        
        return {
            "type": q_type,
            "id": num,
            "question": question,
            "latency_sec": latency,
            "steps_count": len(steps),
            "steps_path": " ➔ ".join(steps),
            "relevance_score": relevance,
            "response": generation,
            "success": True
        }
    except Exception as e:
        latency = time.time() - start_time
        return {
            "type": q_type,
            "id": num,
            "question": question,
            "latency_sec": latency,
            "steps_count": 0,
            "steps_path": f"Error: {str(e)}",
            "relevance_score": 0.0,
            "response": f"Error: {str(e)}",
            "success": False
        }

def generate_evaluation_plots(df: pd.DataFrame):
    """
    Generate comparison charts for Simple vs Complex questions.
    """
    # Calculate averages
    averages = df.groupby("type")[["latency_sec", "steps_count", "relevance_score"]].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Latency Bar Chart
    ax1.bar(averages.index, averages["latency_sec"], color=["#1f6feb", "#58a6ff"], edgecolor="#30363d")
    ax1.set_title("Average Latency Comparison", fontsize=14, color="#58a6ff")
    ax1.set_ylabel("Time (seconds)", fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(averages["latency_sec"]):
        ax1.text(i, v + 0.1, f"{v:.2f}s", ha='center', fontweight='bold')
        
    # 2. Graph Steps Bar Chart
    ax2.bar(averages.index, averages["steps_count"], color=["#2ea44f", "#3fb950"], edgecolor="#30363d")
    ax2.set_title("Average Agent Steps Visited", fontsize=14, color="#58a6ff")
    ax2.set_ylabel("Step Count", fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(averages["steps_count"]):
        ax2.text(i, v + 0.1, f"{v:.1f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plot_path = ASSETS_DIR / "evaluation_charts.png"
    plt.savefig(plot_path, dpi=150, facecolor="#0d1117")
    plt.close()
    
    print(f"Performance charts saved to {plot_path}")
    return plot_path

if __name__ == "__main__":
    # Test script: allows running the evaluation suite from command line
    print("Running automated system evaluation...")
    try:
        run_evaluation_suite()
    except Exception as e:
        print(f"Error executing evaluation: {e}")
