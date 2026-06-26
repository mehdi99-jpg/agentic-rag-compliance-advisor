import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Compliance Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Premium CSS styling ----------
st.markdown("""
<style>
    /* Overall font + spacing */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* App & Sidebar background styling */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stSidebar {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Reduce default Streamlit top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* Header Styling */
    h1, h2, h3, h4, h5, h6 {
        color: #e5e7eb !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600 !important;
    }

    /* Accent highlight color only when necessary */
    .accent-text {
        color: #6366f1 !important;
    }

    /* Card component */
    .status-card {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 14px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22c55e; /* green */
        flex-shrink: 0;
    }

    .status-dot.offline {
        background-color: #ef4444; /* red */
    }

    .section-divider {
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 18px 0;
    }

    /* Main title */
    .app-title {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        font-size: 26px;
        font-weight: 600;
        color: #e5e7eb;
        margin-top: 10px;
        margin-bottom: 4px;
        text-align: center;
    }

    .app-subtitle {
        color: #9ca3af;
        font-size: 14px;
        margin-bottom: 24px;
        text-align: center;
    }

    /* Base chat message container */
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 8px 0;
        margin-bottom: 4px;
    }

    /* The inner bubble - target the message content wrapper */
    [data-testid="stChatMessageContent"] {
        border-radius: 14px;
        padding: 12px 16px;
        font-size: 14.5px;
        line-height: 1.5;
    }

    /* User messages - aligned right, accent background */
    [data-testid="stChatMessage"][data-test-author-name="user"] [data-testid="stChatMessageContent"] {
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.25);
        color: #e5e7eb;
        margin-left: auto;
    }

    /* Assistant messages - left aligned, neutral background */
    [data-testid="stChatMessage"][data-test-author-name="assistant"] [data-testid="stChatMessageContent"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        color: #d1d5db;
    }

    /* Avatar styling - make default avatars smaller/cleaner */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        width: 30px;
        height: 30px;
        border-radius: 8px;
    }

    [data-testid="stChatMessageAvatarUser"] {
        background-color: #6366f1 !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: rgba(255,255,255,0.08) !important;
    }

    /* Expander boxes customization */
    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background-color: #161b22 !important;
        border-radius: 10px !important;
    }

    /* Footer */
    .app-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 10px 0;
        font-size: 12px;
        color: #6b7280;
        background: rgba(13, 17, 23, 0.9);
        border-top: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(4px);
        z-index: 999;
    }

    /* Chat input container restyling */
    [data-testid="stChatInput"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 4px 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.25);
    }

    /* The actual textarea inside */
    [data-testid="stChatInput"] textarea {
        font-size: 15px;
        color: #e5e7eb;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #6b7280;
    }

    /* Send button styling */
    [data-testid="stChatInput"] button {
        background-color: #6366f1 !important;
        border-radius: 10px !important;
        border: none !important;
        transition: background-color 0.15s ease;
    }

    [data-testid="stChatInput"] button:hover {
        background-color: #4f46e5 !important;
    }

    [data-testid="stChatInput"] button svg {
        stroke: white !important;
        fill: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Icon helpers (inline SVG, Lucide-style) ----------
ICONS = {
    "check": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>',
    "folder": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>',
    "bot": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><line x1="8" y1="16" x2="8" y2="16"></line><line x1="16" y1="16" x2="16" y2="16"></line></svg>',
    "zap": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>',
}

# SVG Avatars (Lucide style inline URIs)
ASSISTANT_AVATAR_SVG = """data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%236366f1' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><rect x='3' y='11' width='18' height='10' rx='2'></rect><circle cx='12' cy='5' r='2'></circle><path d='M12 7v4'></path></svg>"""

USER_AVATAR_SVG = """data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='8' r='4'></circle><path d='M4 21v-1a8 8 0 0 1 16 0v1'></path></svg>"""


def status_row(label: str, online: bool = True):
    dot_class = "status-dot" if online else "status-dot offline"
    st.sidebar.markdown(f"""
        <div class="status-card">
            <div class="{dot_class}"></div>
            <span>{label}</span>
        </div>
    """, unsafe_allow_html=True)

def icon_card(icon_key: str, text: str):
    st.sidebar.markdown(f"""
        <div class="status-card">
            {ICONS[icon_key]}
            <span>{text}</span>
        </div>
    """, unsafe_allow_html=True)

# Imports from project modules
from src.vector_store import build_vector_store, CHROMA_DIR
from src.graph import compile_graph

# Setup thread ID session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{int(time.time())}"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize compiled graph agent
if "agent" not in st.session_state:
    try:
        st.session_state.agent = compile_graph()
    except Exception as e:
        st.error(f"Error compiling LangGraph: {e}")

# Title of the Application
st.markdown(f"""
    <div class="app-title">{ICONS['bot']} Agentic RAG Regulatory Compliance Advisor</div>
    <div class="app-subtitle">Powered by LangGraph, Gemini & Groq APIs · Master IIBDCC SMA & IAD Evaluation</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.markdown("##### Configuration & Diagnostics")
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# API Keys Check
st.sidebar.markdown("###### API Connection Status")
api_status = {
    "Gemini API": bool(os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "YOUR_GEMINI_API_KEY_HERE"),
    "Groq API": bool(os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY") != "YOUR_GROQ_API_KEY_HERE"),
    "Tavily API": bool(os.getenv("TAVILY_API_KEY") and os.getenv("TAVILY_API_KEY") != "YOUR_TAVILY_API_KEY_HERE")
}

for name, status in api_status.items():
    status_row(name, online=status)

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# LLM Configuration Selection
st.sidebar.markdown("###### Model Selection")
llm_provider = st.sidebar.selectbox(
    "Primary LLM Brain:",
    options=["Gemini (gemini-2.5-flash)", "Groq (llama-3.1-70b)"],
    index=0
)
os.environ["LLM_PROVIDER"] = "groq" if "Groq" in llm_provider else "gemini"

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Ingestion control
st.sidebar.markdown("###### Vector Database")
if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
    icon_card("folder", "Chroma database is built and loaded")
else:
    st.sidebar.markdown(f"""
        <div class="status-card" style="border-color: rgba(239, 68, 68, 0.2);">
            <div class="status-dot offline"></div>
            <span>Chroma database index not found</span>
        </div>
    """, unsafe_allow_html=True)

if st.sidebar.button("Ingest & Build Database", width='stretch'):
    with st.spinner("Parsing markdown files, chunking, and embedding to Chroma..."):
        try:
            build_vector_store()
            st.sidebar.success("Database built successfully!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to ingest: {e}")

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Graph visualizer
st.sidebar.markdown("###### Graph Architecture Map")
graph_img_path = Path("assets/graph_architecture.png")
if graph_img_path.exists():
    st.sidebar.image(str(graph_img_path), caption="Compiled LangGraph Flow Map", width='stretch')
else:
    st.sidebar.info("Graph visual path not compiled yet. Run a query first.")

# ==========================================
# MAIN CONTAINER TABS
# ==========================================
tab_chat, tab_eval = st.tabs(["Chat Assistant", "Performance Evaluation"])

# TAB 1: CHAT INTERFACE
with tab_chat:
    st.markdown("### Ask any AI regulation question:")
    st.caption("Ask questions about prohibited systems, compliance obligations, or calculate fines based on annual turnover.")
    
    # Render historical conversation
    for msg in st.session_state.messages:
        avatar = ASSISTANT_AVATAR_SVG if isinstance(msg, AIMessage) else USER_AVATAR_SVG
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.content)

    # Input text box
    user_query = st.chat_input("Ex: What are the obligations for a recruitment AI? Calculate the fine if a €50M company violates banned AI regulations.")

    if user_query:
        # Display human message
        with st.chat_message("user", avatar=USER_AVATAR_SVG):
            st.markdown(user_query)
        st.session_state.messages.append(HumanMessage(content=user_query))
        
        # Call agent graph
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        initial_state = {
            "query": user_query,
            "messages": [HumanMessage(content=user_query)],
            "documents": [],
            "generation": "",
            "steps": []
        }
        
        with st.spinner("Agent thinking..."):
            try:
                # Run the LangGraph execution stream
                result = st.session_state.agent.invoke(initial_state, config=config)
                
                # Fetch output from agent state
                generation = result.get("generation", "Sorry, I could not compute a response.")
                steps = result.get("steps", [])
                docs = result.get("documents", [])
                
                # Display assistant response
                with st.chat_message("assistant", avatar=ASSISTANT_AVATAR_SVG):
                    st.markdown(generation)
                st.session_state.messages.append(AIMessage(content=generation))
                
                # Expanders for steps & sources
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("Agentic Execution Steps"):
                        # Format nodes list beautifully
                        formatted_steps = " ➔ ".join([f"`{s.upper()}`" for s in steps])
                        st.markdown(f"**Path taken:**\n{formatted_steps}")
                        st.caption("The agent self-corrects and routes dynamically using conditional edges.")
                with col2:
                    with st.expander("Referenced Sources Cited"):
                        if docs:
                            for idx, doc in enumerate(docs):
                                src = doc.metadata.get('source', 'Unknown')
                                st.markdown(f"**Source {idx+1}:** {src}")
                                st.text_area(label="", value=doc.page_content[:300] + "...", height=100, key=f"src_{idx}")
                        else:
                            st.write("No document sources referenced.")
                            
            except Exception as e:
                st.error(f"Execution Error: {e}")
                st.write("Make sure you have run the database ingestion first.")

# TAB 2: PERFORMANCE EVALUATION DASHBOARD
with tab_eval:
    st.markdown("### System Evaluation Dashboard")
    st.write(
        "Evaluate the performance of your Agentic RAG system. "
        "Running this suite will test the graph on 10 simple questions and 10 complex questions."
    )
    
    # Connect with evaluation module
    eval_btn = st.button("Run Automated Evaluation Suite", type="primary")
    
    if eval_btn:
        with st.spinner("Running 20 tests... (This takes a moment as we test the agent graph)"):
            try:
                # Import evaluation function dynamically to keep loading fast
                from src.evaluation import run_evaluation_suite
                metrics_df, plot_path = run_evaluation_suite()
                
                st.success("Evaluation complete!")
                
                # Show summary
                st.markdown("#### Performance Summary")
                avg_latency = metrics_df["latency_sec"].mean()
                avg_steps = metrics_df["steps_count"].mean()
                relevance_rate = (metrics_df["relevance_score"] >= 0.7).mean() * 100
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Avg Latency", f"{avg_latency:.2f} seconds")
                m2.metric("Avg Graph Steps", f"{avg_steps:.1f} steps")
                m3.metric("Document Relevance Rate", f"{relevance_rate:.1f}%")
                
                # Display plot
                if plot_path.exists():
                    st.image(str(plot_path), caption="Latency & Steps Metrics", width='stretch')
                
                # Display raw data table
                st.markdown("#### Raw Test Results Table")
                st.dataframe(metrics_df[["type", "question", "response", "latency_sec", "steps_count", "relevance_score", "steps_path"]])
                
            except Exception as e:
                st.error(f"Could not run evaluation: {e}")
                st.info("Make sure you have created the `src/evaluation.py` file first.")

# ---------- Footer ----------
st.markdown("""
    <div class="app-footer">
        Developed by <strong>HYNDI ELMEHDI</strong>
    </div>
""", unsafe_allow_html=True)
