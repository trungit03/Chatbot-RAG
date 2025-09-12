import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent))

from rag.chatbot import RAGChatbot
from config import DOCUMENTS_DIR

st.set_page_config(
    page_title="RAG PDF Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_chatbot():
    return RAGChatbot()

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

def main():
    st.markdown('RAG PDF Chatbot')
    st.markdown("**Chat intelligently with your PDF documents!**")
    
    chatbot = init_chatbot()

    with st.sidebar:
        st.header("Document Management")
        
        with st.container():
            st.subheader("Upload PDF files")
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Only support PDF files"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Process file", disabled=st.session_state.processing):
                    if uploaded_files:
                        process_uploaded_files(uploaded_files, chatbot)
                    else:
                        st.warning("Please select PDF file!")
            
            with col2:
                if st.button("Load from Folder"):
                    load_from_directory(chatbot)
        
        st.divider()
        
        st.subheader("Database Info")
        info = chatbot.get_database_info()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Document", info.get('document_count', 0))
        with col2:
            st.metric("Chat History", info.get('chat_history_length', 0))
        
        if info.get('is_initialized', False):
            st.success("Ready to chat!")
        else:
            st.warning("No document available!")
        
        st.divider()
        
        st.subheader("Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear chat"):
                clear_chat_history(chatbot)
        
        with col2:
            if st.button("Clear DB", type="secondary"):
                clear_database(chatbot)
        
        if st.session_state.messages:
            if st.button("Export Chat History"):
                export_chat_history()
        
        st.divider()
        
        # Tips
        with st.expander("Usage Tips"):
            st.markdown("""
            - **Upload PDF**: You can upload multiple PDFs at once  
            - **Ask Questions**: Query the content inside the documents  
            - **Check Sources**: Click "Sources" to see citations  
            - **Export Chat**: Save the conversation to a JSON file
            """)

    if not chatbot.is_initialized:
        st.info("Please upload PDF documents or load from folder to start chatting!")
        
        st.subheader("Sample Questions")
        col1, col2 = st.columns(2)
        with col1:
            st.info("Summarize the main content of the document")
            st.info("Find information about [specific topic]")
        with col2:
            st.info("Analyze data in the document")
            st.info("Provide conclusions from the content")
        return

    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander(f"Sources ({len(message['sources'])} documents)"):
                        for j, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>{j}. {source.get('filename', 'Unknown')}</strong><br>
                                <small>Size: {source.get('file_size', 'N/A')} bytes | 
                                Pages: {source.get('page_count', 'N/A')}</small>
                            </div>
                            """, unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about your PDF documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            sources = chatbot.get_relevant_sources(prompt, top_k=3)
            
            try:
                for chunk in chatbot.stream_chat(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                if sources:
                    with st.expander(f"Source ({len(sources)} documents)"):
                        for i, source in enumerate(sources, 1):
                            relevance = (1 - source.get('distance', 0)) * 100
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>{i}. {source['metadata'].get('filename', 'Unknown')}</strong><br>
                                <small>Relevance: {relevance:.1f}%</small><br>
                                <em>Content: {source['content'][:150]}...</em>
                            </div>
                            """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                full_response = "Sorry, an error occurred while processing your question."
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": [s.get('metadata', {}) for s in sources] if sources else []
        })

def process_uploaded_files(uploaded_files, chatbot):
    st.session_state.processing = True
    
    try:
        with st.spinner("Processing PDF files..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Saving file: {uploaded_file.name}")
                    progress_bar.progress((i + 1) / (len(uploaded_files) * 2))
                    
                    file_path = Path(temp_dir) / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                status_text.text("Analyzing and Processing content...")
                progress_bar.progress(0.75)
                
                success = chatbot.load_documents(temp_dir)
                progress_bar.progress(1.0)
                
                if success:
                    st.success(f"Successfully processed {len(uploaded_files)} file PDF!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to process PDF files!")
    
    except Exception as e:
        st.error(f"Error while processing files: {str(e)}")
    
    finally:
        st.session_state.processing = False

def load_from_directory(chatbot):
    try:
        with st.spinner("Loading from folder..."):
            if DOCUMENTS_DIR.exists() and any(DOCUMENTS_DIR.glob('*.pdf')):
                success = chatbot.load_documents(str(DOCUMENTS_DIR))
                if success:
                    st.success("Successfully loaded documents from folder!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to load documents from folder!")
            else:
                st.warning("No PDF files found in the documents folder!")
    
    except Exception as e:
        st.error(f"Error loading from folder: {str(e)}")

def clear_chat_history(chatbot):
    chatbot.clear_chat_history()
    st.session_state.messages = []
    st.success("Chat history cleared!")
    time.sleep(1)
    st.rerun()

def clear_database(chatbot):
    try:
        chatbot.clear_database()
        st.success("Database cleared!")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")

def export_chat_history():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"
        
        chat_data = {
            "export_time": datetime.now().isoformat(),
            "total_messages": len(st.session_state.messages),
            "messages": st.session_state.messages
        }
        
        json_str = json.dumps(chat_data, ensure_ascii=False, indent=2)
        
        st.download_button(
            label="Download Chat History",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error exporting chat history: {str(e)}")

if __name__ == "__main__":
    main()