"""
Epstein Files Investigative Research Assistant
Streamlit Application
"""
import streamlit as st
import os
from pathlib import Path

from config import Config
from csv_processor import CSVProcessor
from file_search_manager import FileSearchManager
from query_engine import QueryEngine


# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.csv_processor = None
        st.session_state.file_search_manager = None
        st.session_state.query_engine = None
        st.session_state.documents = []
        st.session_state.source_mapping = {}
        st.session_state.upload_complete = False
        st.session_state.chat_history = []


def setup_sidebar():
    """Setup sidebar with configuration and controls"""
    with st.sidebar:
        st.title(f"{Config.APP_ICON} Research Assistant")

        st.markdown("---")

        # Configuration section
        st.subheader("⚙️ Configuration")

        # API Key check
        if Config.GEMINI_API_KEY and Config.GEMINI_API_KEY != 'your_api_key_here':
            st.success("✅ API Key configured")
        else:
            st.error("❌ API Key not configured")
            st.info("Please set GEMINI_API_KEY in your .env file")

        # Model selection
        model = st.selectbox(
            "Model",
            ["gemini-2.5-flash", "gemini-2.5-pro"],
            index=0,
            help="Flash: Faster, Pro: More accurate"
        )
        st.session_state.selected_model = model

        st.markdown("---")

        # Dataset section
        st.subheader("📊 Dataset")

        if st.session_state.csv_processor:
            stats = st.session_state.csv_processor.get_document_stats()
            st.metric("Documents Loaded", stats.get('total_documents', 0))

            # Show categories
            if stats.get('categories'):
                st.markdown("**Categories:**")
                for cat, count in stats['categories'].items():
                    st.write(f"- {cat}: {count}")

        st.markdown("---")

        # Upload section
        st.subheader("📤 Upload Status")

        if st.session_state.upload_complete:
            st.success("✅ Documents uploaded to Gemini")
            if st.session_state.file_search_manager:
                info = st.session_state.file_search_manager.get_store_info()
                st.write(f"Store: {info.get('display_name', 'N/A')}")
        else:
            st.info("ℹ️ Documents not yet uploaded")

        st.markdown("---")

        # Actions
        st.subheader("🔧 Actions")

        if st.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def load_and_process_data():
    """Load and process CSV data"""
    st.header("📁 Step 1: Load Dataset")

    # Check if CSV exists
    csv_path = Config.DATASET_PATH

    if not os.path.exists(csv_path):
        st.error(f"❌ CSV file not found at: {csv_path}")
        st.info("""
        Please place your Epstein dataset CSV file at:
        `data/epstein_dataset.csv`

        Or update the DATASET_PATH in your .env file.
        """)
        return False

    st.success(f"✅ Found CSV: {csv_path}")

    # Load CSV
    if not st.session_state.csv_processor:
        with st.spinner("Loading CSV..."):
            try:
                processor = CSVProcessor(csv_path)
                processor.load_csv()

                # Allow user to limit documents for testing
                total_rows = len(processor.df)
                limit = st.number_input(
                    f"Number of documents to process (max: {total_rows})",
                    min_value=1,
                    max_value=total_rows,
                    value=min(100, total_rows),
                    help="Start with a small number for testing"
                )

                if st.button("📊 Process Documents"):
                    with st.spinner(f"Processing {limit} documents..."):
                        processor.process_documents(limit=limit)
                        st.session_state.csv_processor = processor
                        st.session_state.documents = processor.get_documents()
                        st.session_state.source_mapping = processor.get_source_mapping()
                        st.success(f"✅ Processed {len(st.session_state.documents)} documents")
                        st.rerun()

            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
                return False

    else:
        stats = st.session_state.csv_processor.get_document_stats()
        st.success(f"✅ Loaded {stats['total_documents']} documents")
        return True

    return False


def upload_to_gemini():
    """Upload documents to Gemini File Search"""
    st.header("📤 Step 2: Upload to Gemini File Search")

    if not st.session_state.documents:
        st.warning("⚠️ Please load and process documents first")
        return False

    if st.session_state.upload_complete:
        st.success("✅ Documents already uploaded!")
        return True

    st.info(f"Ready to upload {len(st.session_state.documents)} documents")

    # Export documents as text files first
    if st.button("🚀 Upload to Gemini"):
        try:
            # Initialize File Search Manager
            with st.spinner("Initializing File Search..."):
                manager = FileSearchManager(
                    api_key=Config.get_api_key(),
                    store_name=Config.FILE_SEARCH_STORE_NAME
                )
                manager.create_or_get_store()
                st.session_state.file_search_manager = manager

            # Export documents
            with st.spinner("Exporting documents..."):
                output_dir = "data/processed"
                exported_files = st.session_state.csv_processor.export_documents_as_txt(output_dir)
                st.success(f"✅ Exported {len(exported_files)} files")

            # Upload documents
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total):
                progress_bar.progress(current / total)
                status_text.text(f"Uploading: {current}/{total}")

            stats = manager.upload_documents_batch(
                file_documents=exported_files,
                batch_size=Config.MAX_UPLOAD_BATCH,
                progress_callback=update_progress
            )

            if stats['successful'] > 0:
                st.success(f"✅ Uploaded {stats['successful']} documents successfully!")
                st.session_state.upload_complete = True
                st.rerun()
            else:
                st.error("❌ No documents uploaded successfully")

        except Exception as e:
            st.error(f"Error uploading documents: {str(e)}")
            return False

    return False


def query_interface():
    """Query interface for asking questions"""
    st.header("💬 Step 3: Ask Questions")

    if not st.session_state.upload_complete:
        st.warning("⚠️ Please upload documents first")
        return

    # Initialize query engine
    if not st.session_state.query_engine:
        # Get metadata from file search manager for citations
        metadata_mapping = {}
        if st.session_state.file_search_manager:
            all_metadata = st.session_state.file_search_manager.get_all_metadata()
            # Create mapping from doc_id to source_path
            for display_name, metadata in all_metadata.items():
                doc_id = metadata.get('doc_id')
                source_path = metadata.get('source_path')
                if doc_id and source_path:
                    metadata_mapping[doc_id] = source_path

        st.session_state.query_engine = QueryEngine(
            api_key=Config.get_api_key(),
            model_name=st.session_state.selected_model,
            source_mapping=metadata_mapping
        )

    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - Who are the main individuals mentioned in the documents?
        - What locations appear most frequently?
        - Are there any documents mentioning specific dates or events?
        - What types of communications are included (emails, legal docs, etc.)?
        - Can you summarize documents related to [specific person/topic]?
        """)

    # Query input
    question = st.text_area(
        "Ask a question about the Epstein files:",
        placeholder="e.g., Who appears most frequently in the documents?",
        height=100
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Search", type="primary")

    if ask_button and question:
        with st.spinner("Searching documents..."):
            result = st.session_state.query_engine.query(
                question=question,
                file_search_store=st.session_state.file_search_manager.get_store()
            )

            # Add to chat history
            st.session_state.chat_history.append({
                'question': question,
                'result': result
            })

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📜 Conversation History")

        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**Q{len(st.session_state.chat_history) - i}:** {chat['question']}")

                if chat['result']['success']:
                    st.markdown(chat['result']['answer'])

                    # Display citations
                    citations = chat['result'].get('citations', [])
                    if citations:
                        with st.expander(f"📚 View {len(citations)} Sources"):
                            for j, citation in enumerate(citations, 1):
                                source_path = citation.get('source_path', citation.get('source', 'Unknown'))
                                st.markdown(f"**{j}. {source_path}**")

                                if citation.get('text'):
                                    st.text(citation['text'][:300] + "..." if len(citation['text']) > 300 else citation['text'])
                else:
                    st.error(f"❌ {chat['result']['answer']}")

                st.markdown("---")


def main():
    """Main application"""
    initialize_session_state()

    # Title
    st.title(f"{Config.APP_ICON} Epstein Files Investigative Research Assistant")
    st.markdown("""
    Ask questions about the 25,000+ Epstein document pages using advanced AI search.
    Every answer includes citations with links to original Google Drive sources.
    """)

    # Setup sidebar
    setup_sidebar()

    # Main workflow
    st.markdown("---")

    # Step 1: Load data
    data_loaded = load_and_process_data()

    if data_loaded:
        st.markdown("---")
        # Step 2: Upload
        upload_complete = upload_to_gemini()

        if upload_complete or st.session_state.upload_complete:
            st.markdown("---")
            # Step 3: Query
            query_interface()


if __name__ == "__main__":
    main()
