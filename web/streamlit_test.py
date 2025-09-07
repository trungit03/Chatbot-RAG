import web.streamlit_app as st
import sys
import os
from pathlib import Path
import tempfile
import time

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from rag.chatbot import RAGChatbot
from config import DOCUMENTS_DIR

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = RAGChatbot()
    st.session_state.messages = []
    st.session_state.is_initialized = False


def main():
    st.title("🤖 RAG Chatbot")
    st.markdown("Ask questions about your uploaded documents!")

    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files"
        )

        if uploaded_files:
            if st.button("Process Uploaded Files"):
                process_uploaded_files(uploaded_files)

        st.markdown('---')

        # Load from documents directory
        if st.button("Load from Documents Directory"):
            load_from_directory()

        st.markdown('---')

        # Database info
        st.header("📊 Database Info")
        info = st.session_state.chatbot.get_database_info()
        st.metric("Documents", info.get('document_count', 0))
        st.metric("Chat History", info.get('chat_history_length', 0))

        if info.get('is_initialized', False):
            st.success("✅ Ready to chat!")
        else:
            st.warning("⚠️ No documents loaded")

        st.markdown('---')

        # Actions
        st.header("🔧 Actions")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Clear Chat"):
                st.session_state.chatbot.clear_chat_history()
                st.session_state.messages = []
                st.success("Chat cleared!")

        with col2:
            if st.button("Clear Database"):
                st.session_state.chatbot.clear_database()
                st.session_state.is_initialized = False
                st.success("Database cleared!")

    # Main chat interface
    if not st.session_state.chatbot.is_initialized:
        st.info("👆 Please upload documents or load from the documents directory to start chatting!")
        return

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**{i}. {source.get('filename', 'Unknown')}**")
                        if 'distance' in source:
                            st.write(f"Relevance: {1 - source['distance']:.2%}")

    # Chat input
    if prompt := st.text_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Get relevant sources
            sources = st.session_state.chatbot.get_relevant_sources(prompt, top_k=3)

            # Stream the response
            for chunk in st.session_state.chatbot.stream_chat(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Show sources
            if sources:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(sources, 1):
                        st.write(f"**{i}. {source['metadata'].get('filename', 'Unknown')}**")
                        if 'distance' in source:
                            st.write(f"Relevance: {1 - source['distance']:.2%}")
                        st.write(f"Content preview: {source['content'][:200]}...")
                        st.markdown('---')

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": [s['metadata'] for s in sources]
        })


def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    try:
        with st.spinner("Processing uploaded files..."):
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files
                for uploaded_file in uploaded_files:
                    file_path = Path(temp_dir) / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                # Load documents
                success = st.session_state.chatbot.load_documents(temp_dir)

                if success:
                    st.session_state.is_initialized = True
                    st.success(f"✅ Successfully processed {len(uploaded_files)} files!")
                else:
                    st.error("❌ Failed to process files!")

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")


def load_from_directory():
    """Load documents from the default directory"""
    try:
        with st.spinner("Loading documents from directory..."):
            if not DOCUMENTS_DIR.exists():
                st.error(f"Documents directory not found: {DOCUMENTS_DIR}")
                return

            success = st.session_state.chatbot.load_documents(str(DOCUMENTS_DIR))

            if success:
                st.session_state.is_initialized = True
                st.success("✅ Documents loaded successfully!")
            else:
                st.error("❌ No documents found or failed to load!")

    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")


if __name__ == "__main__":
    main()