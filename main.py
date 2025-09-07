import argparse
import sys
from pathlib import Path
import logging

from rag.chatbot import RAGChatbot
from config import DOCUMENTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot")
    parser.add_argument(
        "--web",
        choices=["streamlit", "gradio"],
        help="Launch web interface"
    )
    parser.add_argument(
        "--load-docs",
        type=str,
        help="Path to documents to load"
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear the vector database"
    )

    args = parser.parse_args()

    if args.web:
        if args.web == "streamlit":
            import subprocess
            subprocess.run(["streamlit", "run", "web/streamlit_app.py"])
        elif args.web == "gradio":
            import subprocess
            subprocess.run(["python", "web/gradio_app.py"])
        return

    chatbot = RAGChatbot()

    if args.clear_db:
        print("Clearing vector database...")
        chatbot.clear_database()
        print("Database cleared!")
        return

    if args.load_docs:
        print(f"Loading documents from: {args.load_docs}")
        success = chatbot.load_documents(args.load_docs)
        if success:
            print("Documents loaded successfully!")
        else:
            print("Failed to load documents!")
            return
    else:
        if DOCUMENTS_DIR.exists() and any(DOCUMENTS_DIR.iterdir()):
            print(f"Loading documents from default directory: {DOCUMENTS_DIR}")
            success = chatbot.load_documents(str(DOCUMENTS_DIR))
            if not success:
                print("Failed to load documents from default directory!")
        else:
            print(f"No documents found in {DOCUMENTS_DIR}")
            print("Please add documents to the documents directory or specify a path with --load-docs")
            return

    print("\n" + "=" * 50)
    print("RAG Chatbot - Interactive Mode")
    print("=" * 50)
    print("Type 'quit' or 'exit' to stop")
    print("Type 'clear' to clear chat history")
    print("Type 'info' to see database information")
    print("Type 'sources <query>' to see relevant sources")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break

            if user_input.lower() == 'clear':
                chatbot.clear_chat_history()
                print("Chat history cleared!")
                continue

            if user_input.lower() == 'info':
                info = chatbot.get_database_info()
                print(f"\nDatabase Info:")
                print(f"- Collection: {info.get('name', 'N/A')}")
                print(f"- Documents: {info.get('document_count', 0)}")
                print(f"- Initialized: {info.get('is_initialized', False)}")
                print(f"- Chat History: {info.get('chat_history_length', 0)} exchanges")
                continue

            if user_input.lower().startswith('sources '):
                query = user_input[8:]  
                sources = chatbot.get_relevant_sources(query, top_k=3)
                print(f"\nRelevant sources for: '{query}'")
                for i, source in enumerate(sources, 1):
                    print(f"{i}. {source['metadata'].get('filename', 'Unknown')}")
                    print(f"   Distance: {source.get('distance', 'N/A'):.4f}")
                    print(f"   Content preview: {source['content'][:100]}...")
                continue

            if not user_input:
                continue

            print("\nBot: ", end="", flush=True)

            for chunk in chatbot.stream_chat(user_input):
                print(chunk, end="", flush=True)
            print()  

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()