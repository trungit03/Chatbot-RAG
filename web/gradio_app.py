import gradio as gr
import sys
import os
from pathlib import Path
import tempfile
import json

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from rag.chatbot import RAGChatbot
from config import DOCUMENTS_DIR

# Global chatbot instance
chatbot = RAGChatbot()


def process_files(files):
    """Process uploaded files"""
    if not files:
        return "No files uploaded.", ""

    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            for file in files:
                file_path = Path(temp_dir) / Path(file.name).name
                # Copy file to temp directory
                import shutil
                shutil.copy2(file.name, file_path)

            # Load documents
            success = chatbot.load_documents(temp_dir)

            if success:
                info = chatbot.get_database_info()
                return f"✅ Successfully processed {len(files)} files! Database now contains {info.get('document_count', 0)} documents.", get_database_info()
            else:
                return "❌ Failed to process files!", get_database_info()

    except Exception as e:
        return f"Error processing files: {str(e)}", get_database_info()


def load_from_directory():
    """Load documents from default directory"""
    try:
        if not DOCUMENTS_DIR.exists():
            return f"❌ Documents directory not found: {DOCUMENTS_DIR}", get_database_info()

        success = chatbot.load_documents(str(DOCUMENTS_DIR))

        if success:
            info = chatbot.get_database_info()
            return f"✅ Documents loaded successfully! Database contains {info.get('document_count', 0)} documents.", get_database_info()
        else:
            return "❌ No documents found or failed to load!", get_database_info()

    except Exception as e:
        return f"Error loading documents: {str(e)}", get_database_info()


def get_database_info():
    """Get database information as formatted string"""
    info = chatbot.get_database_info()
    return f"""
**Database Information:**
- Collection: {info.get('name', 'N/A')}
- Document Count: {info.get('document_count', 0)}
- Initialized: {'✅ Yes' if info.get('is_initialized', False) else '❌ No'}
- Chat History: {info.get('chat_history_length', 0)} exchanges
"""


def chat_with_bot(message, history):
    """Chat with the bot"""
    if not chatbot.is_initialized:
        return history + [[message, "Please upload documents first before asking questions."]]

    try:
        # Get response from chatbot
        response = chatbot.chat(message)
        return history + [[message, response]]
    except Exception as e:
        return history + [[message, f"Error: {str(e)}"]]


def clear_chat():
    """Clear chat history"""
    chatbot.clear_chat_history()
    return []


def clear_database():
    """Clear the database"""
    chatbot.clear_database()
    return "Database cleared!", get_database_info()


def get_sources(query):
    """Get relevant sources for a query"""
    if not chatbot.is_initialized:
        return "Please upload documents first."

    try:
        sources = chatbot.get_relevant_sources(query, top_k=5)
        if not sources:
            return "No relevant sources found."

        result = f"**Relevant sources for: '{query}'**\n\n"
        for i, source in enumerate(sources, 1):
            result += f"**{i}. {source['metadata'].get('filename', 'Unknown')}**\n"
            result += f"Relevance: {1 - source.get('distance', 0):.2%}\n"
            result += f"Content preview: {source['content'][:200]}...\n\n"

        return result
    except Exception as e:
        return f"Error getting sources: {str(e)}"


# Create Gradio interface
def create_interface():
    with gr.Blocks(title="RAG Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 RAG Chatbot")
        gr.Markdown("Upload documents and ask questions about their content!")

        with gr.Tab("💬 Chat"):
            chatbot_interface = gr.Chatbot(
                label="Chat with your documents",
                height=400,
                show_label=True
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Ask a question about your documents...",
                    scale=4
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary")

            # Chat functionality
            msg.submit(chat_with_bot, [msg, chatbot_interface], [chatbot_interface])
            msg.submit(lambda: "", None, [msg])
            send_btn.click(chat_with_bot, [msg, chatbot_interface], [chatbot_interface])
            send_btn.click(lambda: "", None, [msg])
            clear_btn.click(clear_chat, None, [chatbot_interface])

        with gr.Tab("📁 Document Management"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Upload Documents")
                    file_upload = gr.Files(
                        label="Upload PDF, TXT, or DOCX files",
                        file_types=[".pdf", ".txt", ".docx"],
                        file_count="multiple"
                    )
                    upload_btn = gr.Button("Process Files", variant="primary")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)

                with gr.Column():
                    gr.Markdown("### Load from Directory")
                    load_btn = gr.Button("Load from Documents Directory", variant="secondary")
                    load_status = gr.Textbox(label="Load Status", interactive=False)

            with gr.Row():
                database_info = gr.Markdown(get_database_info())

            with gr.Row():
                clear_db_btn = gr.Button("Clear Database", variant="stop")
                clear_db_status = gr.Textbox(label="Clear Status", interactive=False)

            # Document management functionality
            upload_btn.click(
                process_files,
                [file_upload],
                [upload_status, database_info]
            )
            load_btn.click(
                load_from_directory,
                None,
                [load_status, database_info]
            )
            clear_db_btn.click(
                clear_database,
                None,
                [clear_db_status, database_info]
            )

        with gr.Tab("🔍 Source Explorer"):
            gr.Markdown("### Find Relevant Sources")
            gr.Markdown("Enter a query to see which document sections are most relevant.")

            with gr.Row():
                source_query = gr.Textbox(
                    label="Query",
                    placeholder="Enter your question to find relevant sources...",
                    scale=4
                )
                source_btn = gr.Button("Find Sources", scale=1, variant="primary")

            sources_output = gr.Markdown(label="Relevant Sources")

            source_btn.click(get_sources, [source_query], [sources_output])
            source_query.submit(get_sources, [source_query], [sources_output])

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## RAG Chatbot

            This is a Retrieval-Augmented Generation (RAG) chatbot that allows you to:

            - **Upload Documents**: Support for PDF, TXT, and DOCX files
            - **Ask Questions**: Get answers based on your document content
            - **View Sources**: See which parts of your documents were used to answer questions
            - **Manage Database**: Clear and reload your document database

            ### How it works:
            1. Upload your documents or load from the documents directory
            2. The system processes and indexes your documents
            3. Ask questions in natural language
            4. Get answers based on the content of your documents

            ### Features:
            - **Vector Search**: Uses embeddings to find relevant document sections
            - **Local LLM**: Powered by Ollama for privacy and control
            - **Multiple Formats**: Supports various document types
            - **Source Attribution**: Shows which documents were used for answers

            ### Requirements:
            - Ollama must be running locally
            - Documents should be in supported formats (PDF, TXT, DOCX)
            """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )