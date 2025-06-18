import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
import chromadb
import nltk
import pandas as pd

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Page configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Detect dark mode
is_dark_theme = st.sidebar.checkbox("Dark Theme", value=True)

# Custom CSS with better visibility and contrast
st.markdown(f"""
<style>
    /* Base theme */
    .main {{
        background-color: {("#1E1E1E" if is_dark_theme else "#f5f7f9")};
        color: {("#FFFFFF" if is_dark_theme else "#333333")};
    }}
    .stApp {{
        max-width: 1200px;
        margin: 0 auto;
    }}
    
    /* Header styling */
    .header {{
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background-color: {("#2D2D2D" if is_dark_theme else "#FFFFFF")};
        border-radius: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    .header-icon {{
        font-size: 2.5rem;
        margin-right: 1rem;
        color: {("#4D8BF9" if is_dark_theme else "#2b79e4")};
    }}
    .header-text {{
        flex-grow: 1;
    }}
    .header-text h1 {{
        margin: 0;
        color: {("#FFFFFF" if is_dark_theme else "#333333")};
        font-size: 2rem;
    }}
    .header-text p {{
        margin: 0;
        color: {("#CCCCCC" if is_dark_theme else "#666666")};
    }}
    
    /* Chat message styling */
    .chat-message {{
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    .chat-message.user {{
        background-color: {("#2C3E50" if is_dark_theme else "#e6f3ff")};
        border-left: 5px solid {("#3498DB" if is_dark_theme else "#2b79e4")};
    }}
    .chat-message.bot {{
        background-color: {("#2D2D2D" if is_dark_theme else "#FFFFFF")};
        border-left: 5px solid {("#2ECC71" if is_dark_theme else "#42b983")};
    }}
    .chat-message .avatar {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-right: 1rem;
    }}
    .chat-message .message {{
        width: 100%;
        color: {("#FFFFFF" if is_dark_theme else "#333333")};
        font-size: 1rem;
        line-height: 1.5;
    }}
    
    /* Citation styling */
    .citation {{
        background-color: {("#3A3A3A" if is_dark_theme else "#fff8e1")};
        border-left: 3px solid {("#F39C12" if is_dark_theme else "#ffc107")};
        padding: 0.7rem 1rem;
        margin-top: 0.7rem;
        font-size: 0.9rem;
        border-radius: 0.25rem;
        color: {("#E0E0E0" if is_dark_theme else "#333333")};
    }}
    .citation-icon {{
        margin-right: 0.5rem;
        color: {("#F39C12" if is_dark_theme else "#ffa000")};
    }}
    .document-icon {{
        font-size: 1.2rem;
        margin-right: 0.5rem;
        color: {("#E0E0E0" if is_dark_theme else "#455a64")};
    }}
    
    /* Button styling */
    .stButton>button {{
        background-color: {("#4D8BF9" if is_dark_theme else "#2b79e4")};
        color: {("#FFFFFF" if is_dark_theme else "#FFFFFF")};
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .stButton>button:hover {{
        background-color: {("#3A7CE5" if is_dark_theme else "#1a66c9")};
    }}
    
    /* Data stats cards */
    .data-stats {{
        background-color: {("#2D2D2D" if is_dark_theme else "#FFFFFF")};
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    .data-stats h3 {{
        margin-top: 0;
        color: {("#E0E0E0" if is_dark_theme else "#333333")};
        font-size: 1.2rem;
    }}
    .data-stats p {{
        margin-bottom: 0;
        color: {("#CCCCCC" if is_dark_theme else "#666666")};
    }}
    .data-stats strong {{
        color: {("#4D8BF9" if is_dark_theme else "#2b79e4")};
    }}
    
    /* Document browser styling */
    .document-viewer {{
        background-color: {("#2D2D2D" if is_dark_theme else "#FFFFFF")};
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        border-left: 5px solid {("#4D8BF9" if is_dark_theme else "#2b79e4")};
        max-height: 400px;
        overflow-y: auto;
    }}
    .document-viewer h4 {{
        margin-top: 0;
        color: {("#E0E0E0" if is_dark_theme else "#333333")};
    }}
    .document-viewer pre {{
        white-space: pre-wrap;
        font-family: 'Courier New', monospace;
        background-color: {("#1E1E1E" if is_dark_theme else "#f5f5f5")};
        padding: 1rem;
        border-radius: 0.25rem;
        color: {("#E0E0E0" if is_dark_theme else "#333333")};
        font-size: 0.9rem;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {("#2D2D2D" if is_dark_theme else "#f5f7f9")};
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
        color: {("#CCCCCC" if is_dark_theme else "#666666")};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {("#4D8BF9" if is_dark_theme else "#2b79e4")};
        color: {("#FFFFFF" if is_dark_theme else "#FFFFFF")};
    }}
    
    /* File uploader styling */
    .css-1aehpvj {{
        color: {("#E0E0E0" if is_dark_theme else "#333333")} !important;
    }}
    
    /* Chat input styling */
    .stChatInput {{
        background-color: {("#2D2D2D" if is_dark_theme else "#FFFFFF")};
        border-radius: 0.5rem;
        padding: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    .stChatInput input {{
        color: {("#E0E0E0" if is_dark_theme else "#333333")};
        background-color: {("#3A3A3A" if is_dark_theme else "#f5f7f9")};
        padding: 0.75rem 1rem;
        border-radius: 0.25rem;
        border: 1px solid {("#555555" if is_dark_theme else "#dddddd")};
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: {("#1E1E1E" if is_dark_theme else "#f5f7f9")};
    }}
    .css-1lcbmhc {{
        color: {("#E0E0E0" if is_dark_theme else "#333333")};
    }}
    
    /* Ensure dataframe visibility */
    .dataframe {{
        color: {("#E0E0E0" if is_dark_theme else "#333333")};
    }}
    .dataframe th {{
        background-color: {("#3A3A3A" if is_dark_theme else "#f5f7f9")};
        color: {("#E0E0E0" if is_dark_theme else "#333333")};
        font-weight: 600;
        padding: 0.5rem;
        border: 1px solid {("#555555" if is_dark_theme else "#dddddd")};
    }}
    .dataframe td {{
        background-color: {("#2D2D2D" if is_dark_theme else "#FFFFFF")};
        color: {("#E0E0E0" if is_dark_theme else "#333333")};
        padding: 0.5rem;
        border: 1px solid {("#555555" if is_dark_theme else "#dddddd")};
    }}
    
    /* Footer styling */
    .footer {{
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        color: {("#CCCCCC" if is_dark_theme else "#6c757d")};
        font-size: 0.8rem;
        border-top: 1px solid {("#555555" if is_dark_theme else "#dddddd")};
    }}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_engine' not in st.session_state:
    st.session_state.chat_engine = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Sidebar with improved visibility
with st.sidebar:
    st.markdown(f"<h2 style='color: {'#E0E0E0' if is_dark_theme else '#333333'};'>📚 Document Q&A System</h2>", unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color: {'#555555' if is_dark_theme else '#dddddd'};'>", unsafe_allow_html=True)

    # Model selection with better labeling
    st.markdown(f"<h3 style='color: {'#E0E0E0' if is_dark_theme else '#333333'}; margin-top: 20px;'>🤖 Model Configuration</h3>", unsafe_allow_html=True)
    llm_model = st.selectbox(
        "Select LLM Model",
        options=["llama3-groq-tool-use:8b", "qwen2.5:7b-instruct-q4_K_M"],
        index=0
    )
    # Document upload with clearer visuals
    st.markdown(f"<h3 style='color: {'#E0E0E0' if is_dark_theme else '#333333'}; margin-top: 20px;'>📄 Document Management</h3>", unsafe_allow_html=True)
    
    # Initialize button with better visibility
    initialize_button = st.button("Initialize System", use_container_width=True)
    
    if initialize_button:
        with st.spinner("Initializing document Q&A system..."):
            try:
                # Initialize AI Model & Embeddings
                pre_llm = Ollama(model=llm_model, request_timeout=600)
                embed_model = OllamaEmbedding(model_name='nomic-embed-text')
                
                # Set global settings for LlamaIndex
                Settings.llm = pre_llm
                Settings.embed_model = embed_model
                
                # Initialize ChromaDB Client and Collection
                chroma_client = chromadb.PersistentClient(path="./chroma_db_IonIdea_new")
                collection_name = "document_knowledge_base"
                collection = chroma_client.get_or_create_collection(collection_name)
                st.session_state.collection = collection
                # Initialize Chat Memory with Token Limit
                chat_memory = ChatMemoryBuffer(token_limit=2048)
                
                # Document Ingestion and Index Creation
                nodes = []
                st.info("Loading index from ChromaDB...")
                # Load stored documents from ChromaDB
                stored_data = collection.get()
                
                for doc, meta in zip(stored_data["documents"], stored_data["metadatas"]):
                    filename = meta.get("file_name", "Unknown File")
                    page = meta.get("page_label", "Unknown Page")
                    
                    # Convert raw text and metadata back into a Document
                    document = Document(text=doc, metadata={"file_name": filename, "page_label": page})
                    nodes.append(document)
                
                # Create the index from stored documents
                index = VectorStoreIndex.from_documents(nodes, show_progress=True)
                st.session_state.index = index
                
                st.success(f"Loaded {len(nodes)} document chunks from ChromaDB.")
                
                system_prompt = (
                    "You are a chatbot fine-tuned by Ion Idea for answering questions regarding Ion Idea. "
                    "You are to answer any questions related only to the Ion Idea company, their work, or any of their subsidiaries "
                    "by retrieving the information from a knowledge base. "
                    "If the retrieved information does not contain what the user asked, you must say: 'I don't know.' "
                    "There should be zero hallucinations and no fabricated answers—only facts. "
                    "You must never mention the file names in your responses. "
                    "If the user is engaging in very basic and casual chat, you are allowed to answer. "
                    "The moment the user goes out of context, you must refuse to answer any non-casual questions that are not "
                    "covered in the retrieved content and not related to Ion Idea or its subsidiaries."
                )

                
                # Setup the Retriever and Chat Engine
                retriever = index.as_retriever(similarity_top_k=5, embed_model=embed_model)
                
                chat_engine = CondensePlusContextChatEngine.from_defaults(
                    llm=pre_llm,
                    retriever=retriever,
                    memory=chat_memory,
                    system_prompt=system_prompt,
                )
                
                st.session_state.chat_engine = chat_engine
                st.session_state.initialized = True
                
            except Exception as e:
                st.error(f"Error initializing system: {str(e)}")

# Main content with improved header
st.markdown("""
<div class="header">
    <div class="header-icon">📚</div>
    <div class="header-text">
        <h1>Document Q&A System</h1>
        <p>Ask questions about your documents and get accurate answers with citations</p>
    </div>
</div>
""", unsafe_allow_html=True)

# System stats with better visibility
if st.session_state.initialized and st.session_state.collection:
    collection_count = st.session_state.collection.count()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="data-stats">
            <h3>📊 Document Stats</h3>
            <p>Indexed chunks: <strong>{collection_count}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="data-stats">
            <h3>🔍 Search Configuration</h3>
            <p>Top-K results: <strong>5</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="data-stats">
            <h3>🤖 Model</h3>
            <p>Using: <strong>{llm_model.split('/')[-1]}</strong></p>
        </div>
        """, unsafe_allow_html=True)

# Document browser tab with improved visibility
tab1, tab2 = st.tabs(["💬 Chat", "📑 Document Browser"])

with tab1:
    # Clear chat option
    if st.session_state.messages:
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.messages = []
            st.experimental_rerun()
    
    # Display chat messages with improved visibility
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="avatar">👤</div>
                    <div class="message">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                response_content = message["content"]
                citations = message.get("citations", [])
                
                citations_html = ""
                if citations:
                    citations_html = "<div class='citation'><span class='citation-icon'></span> <strong>Citations:</strong><br>"
                    for citation in citations:
                        citations_html += f"<span class='document-icon'>📄</span> {citation}<br>"
                    citations_html += "</div>"
                
                st.markdown(f"""
                <div class="chat-message bot">
                    <div class="avatar">🤖</div>
                    <div class="message">
                        {response_content}
                        {citations_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input for user query with improved visibility
    user_query = st.chat_input("Ask a question about your documents...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message immediately
        with st.container():
            st.markdown(f"""
            <div class="chat-message user">
                <div class="avatar">👤</div>
                <div class="message">{user_query}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Check if system is initialized
        if not st.session_state.initialized:
            error_msg = "Please initialize the system first using the button in the sidebar."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            with st.container():
                st.markdown(f"""
                <div class="chat-message bot">
                    <div class="avatar">🤖</div>
                    <div class="message">{error_msg}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Show loading spinner
            with st.spinner("Thinking..."):
                try:
                    # Get response from chat engine
                    response = st.session_state.chat_engine.chat(user_query)
                    
                    # Collect unique citations
                    unique_citations = set()
                    if response.source_nodes:
                        for node in response.source_nodes:
                            meta = node.metadata
                            file_name = meta.get("file_name", "Unknown File")
                            page_label = meta.get("page_label", "Unknown Page")
                            unique_citations.add(f"{file_name}, Page: {page_label}")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.response, 
                        "citations": list(unique_citations)
                    })
                    
                    # Display assistant message immediately
                    with st.container():
                        citations_html = ""
                        if unique_citations:
                            citations_html = "<div class='citation'><span class='citation-icon'>📄</span> <strong>Citations:</strong><br>"
                            for citation in unique_citations:
                                citations_html += f"<span class='document-icon'>📄</span> {citation}<br>"
                            citations_html += "</div>"
                        
                        st.markdown(f"""
                        <div class="chat-message bot">
                            <div class="avatar">🤖</div>
                            <div class="message">
                                {response.response}
                                {citations_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="chat-message bot">
                            <div class="avatar">🤖</div>
                            <div class="message">{error_msg}</div>
                        </div>
                        """, unsafe_allow_html=True)

with tab2:
    if st.session_state.initialized and st.session_state.collection:
        stored_data = st.session_state.collection.get()
        
        if stored_data and len(stored_data["documents"]) > 0:
            # Create a DataFrame for easy display
            df_data = []
            for i, (doc, meta, doc_id) in enumerate(zip(stored_data["documents"], stored_data["metadatas"], stored_data["ids"])):
                filename = meta.get("file_name", "Unknown File")
                page = meta.get("page_label", "Unknown Page")
                preview = doc[:100] + "..." if len(doc) > 100 else doc
                df_data.append({"ID": doc_id, "Filename": filename, "Page": page, "Preview": preview})
            
            df = pd.DataFrame(df_data)
            
            # Add search and filter with better visibility
            st.markdown(f"<h3 style='color: {'#E0E0E0' if is_dark_theme else '#333333'};'>🔍 Search Documents</h3>", unsafe_allow_html=True)
            search_term = st.text_input("Search for text in documents")
            
            if search_term:
                filtered_df = df[df["Preview"].str.contains(search_term, case=False, na=False)]
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
            
            # Document viewer with better visibility
            if df_data:
                st.markdown(f"<h3 style='color: {'#E0E0E0' if is_dark_theme else '#333333'};'>📄 Document Viewer</h3>", unsafe_allow_html=True)
                selected_doc = st.selectbox("Select document to view", options=df["ID"].tolist(), format_func=lambda x: f"{df[df['ID'] == x]['Filename'].iloc[0]} - Page {df[df['ID'] == x]['Page'].iloc[0]}")
                
                if selected_doc:
                    doc_index = stored_data["ids"].index(selected_doc)
                    document_content = stored_data["documents"][doc_index]
                    
                    st.markdown(f"""
                    <div class="document-viewer">
                        <h4>Document Content</h4>
                        <pre>{document_content}</pre>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No documents found in the collection. Please upload and index documents first.")
    else:
        st.warning("Please initialize the system first using the button in the sidebar.")

# Footer with better visibility
st.markdown(f"""
<div class="footer">
    <p>Document Q&A System powered by LlamaIndex and Together AI</p>
</div>
""", unsafe_allow_html=True)