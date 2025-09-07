import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime
import json

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from rag.chatbot import RAGChatbot
from config import DOCUMENTS_DIR

# Page config
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #1f77b4;
}
.source-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
    margin-bottom: 0.5rem;
}
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    text-align: center;
}
.success-message {
    background-color: #d4edda;
    color: #155724;
    padding: 0.75rem;
    border-radius: 0.25rem;
    border: 1px solid #c3e6cb;
}
.warning-message {
    background-color: #fff3cd;
    color: #856404;
    padding: 0.75rem;
    border-radius: 0.25rem;
    border: 1px solid #ffeaa7;
}
.error-message {
    background-color: #f8d7da;
    color: #721c24;
    padding: 0.75rem;
    border-radius: 0.25rem;
    border: 1px solid #f5c6cb;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def init_chatbot():
    return RAGChatbot()

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 RAG PDF Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("**Trò chuyện thông minh với tài liệu PDF của bạn!**")
    
    # Initialize chatbot
    chatbot = init_chatbot()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Quản lý tài liệu")
        
        # File upload section
        with st.container():
            st.subheader("📤 Tải lên PDF")
            uploaded_files = st.file_uploader(
                "Chọn file PDF",
                type=['pdf'],
                accept_multiple_files=True,
                help="Chỉ hỗ trợ file PDF"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Xử lý file", disabled=st.session_state.processing):
                    if uploaded_files:
                        process_uploaded_files(uploaded_files, chatbot)
                    else:
                        st.warning("Vui lòng chọn file PDF!")
            
            with col2:
                if st.button("📂 Tải từ thư mục"):
                    load_from_directory(chatbot)
        
        st.divider()
        
        # Database info
        st.subheader("📊 Thông tin cơ sở dữ liệu")
        info = chatbot.get_database_info()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Tài liệu", info.get('document_count', 0))
        with col2:
            st.metric("💬 Lịch sử chat", info.get('chat_history_length', 0))
        
        # Status indicator
        if info.get('is_initialized', False):
            st.success("✅ Sẵn sàng trò chuyện!")
        else:
            st.warning("⚠️ Chưa có tài liệu nào")
        
        st.divider()
        
        # Actions
        st.subheader("🛠️ Thao tác")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Xóa chat"):
                clear_chat_history(chatbot)
        
        with col2:
            if st.button("💥 Xóa DB", type="secondary"):
                clear_database(chatbot)
        
        # Export chat history
        if st.session_state.messages:
            if st.button("📥 Xuất lịch sử"):
                export_chat_history()
        
        st.divider()
        
        # Tips
        with st.expander("💡 Mẹo sử dụng"):
            st.markdown("""
            - **Tải PDF**: Chọn nhiều file PDF cùng lúc
            - **Đặt câu hỏi**: Hỏi về nội dung trong tài liệu
            - **Xem nguồn**: Click vào "Sources" để xem trích dẫn
            - **Xuất chat**: Lưu cuộc trò chuyện thành file JSON
            """)

    # Main chat interface
    if not chatbot.is_initialized:
        st.info("👆 Vui lòng tải lên tài liệu PDF hoặc tải từ thư mục để bắt đầu trò chuyện!")
        
        # Show sample questions
        st.subheader("🤔 Câu hỏi mẫu")
        col1, col2 = st.columns(2)
        with col1:
            st.info("📝 Tóm tắt nội dung chính của tài liệu")
            st.info("🔍 Tìm thông tin về [chủ đề cụ thể]")
        with col2:
            st.info("📊 Phân tích dữ liệu trong tài liệu")
            st.info("💡 Đưa ra kết luận từ nội dung")
        return

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander(f"📚 Nguồn tham khảo ({len(message['sources'])} tài liệu)"):
                        for j, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>📄 {j}. {source.get('filename', 'Unknown')}</strong><br>
                                <small>📏 Kích thước: {source.get('file_size', 'N/A')} bytes | 
                                📑 Trang: {source.get('page_count', 'N/A')}</small>
                            </div>
                            """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Hỏi về tài liệu PDF của bạn..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Get relevant sources first
            sources = chatbot.get_relevant_sources(prompt, top_k=3)
            
            # Stream the response
            try:
                for chunk in chatbot.stream_chat(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Show sources
                if sources:
                    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} tài liệu)"):
                        for i, source in enumerate(sources, 1):
                            relevance = (1 - source.get('distance', 0)) * 100
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>📄 {i}. {source['metadata'].get('filename', 'Unknown')}</strong><br>
                                <small>🎯 Độ liên quan: {relevance:.1f}%</small><br>
                                <em>Nội dung: {source['content'][:150]}...</em>
                            </div>
                            """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Lỗi khi tạo phản hồi: {str(e)}")
                full_response = "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn."
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": [s.get('metadata', {}) for s in sources] if sources else []
        })

def process_uploaded_files(uploaded_files, chatbot):
    """Process uploaded PDF files"""
    st.session_state.processing = True
    
    try:
        with st.spinner("🔄 Đang xử lý file PDF..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Đang lưu file: {uploaded_file.name}")
                    progress_bar.progress((i + 1) / (len(uploaded_files) * 2))
                    
                    file_path = Path(temp_dir) / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Load documents
                status_text.text("Đang xử lý và phân tích nội dung...")
                progress_bar.progress(0.75)
                
                success = chatbot.load_documents(temp_dir)
                progress_bar.progress(1.0)
                
                if success:
                    st.success(f"✅ Đã xử lý thành công {len(uploaded_files)} file PDF!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Không thể xử lý file PDF!")
    
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý file: {str(e)}")
    
    finally:
        st.session_state.processing = False

def load_from_directory(chatbot):
    """Load documents from default directory"""
    try:
        with st.spinner("📂 Đang tải từ thư mục..."):
            if DOCUMENTS_DIR.exists() and any(DOCUMENTS_DIR.glob('*.pdf')):
                success = chatbot.load_documents(str(DOCUMENTS_DIR))
                if success:
                    st.success("✅ Đã tải tài liệu từ thư mục thành công!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Không thể tải tài liệu từ thư mục!")
            else:
                st.warning("⚠️ Không tìm thấy file PDF trong thư mục documents!")
    
    except Exception as e:
        st.error(f"❌ Lỗi khi tải từ thư mục: {str(e)}")

def clear_chat_history(chatbot):
    """Clear chat history"""
    chatbot.clear_chat_history()
    st.session_state.messages = []
    st.success("🗑️ Đã xóa lịch sử chat!")
    time.sleep(1)
    st.rerun()

def clear_database(chatbot):
    """Clear vector database"""
    try:
        chatbot.clear_database()
        st.success("💥 Đã xóa cơ sở dữ liệu!")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"❌ Lỗi khi xóa database: {str(e)}")

def export_chat_history():
    """Export chat history to JSON"""
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
            label="📥 Tải xuống lịch sử chat",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"❌ Lỗi khi xuất lịch sử: {str(e)}")

if __name__ == "__main__":
    main()