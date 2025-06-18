"""
Document to Markdown Converter - Main Application
Updated with Enterprise LLM Integration
"""

import streamlit as st
import os

# UI Components
from src.ui.components import setup_page_config
from src.ui.sidebar import setup_enhanced_sidebar
from src.ui.main_content import render_welcome_section
from src.ui.output_display import display_enhanced_output_section
from src.ui.folder_results import display_folder_results
from src.ui.about_tab import render_about_tab

# Enterprise LLM Integration
try:
    from enterprise_file_converter import (
        convert_file_to_markdown_enterprise,
        process_folder_enterprise,
        get_enterprise_llm_status
    )

    ENTERPRISE_LLM_AVAILABLE = True
except ImportError:
    ENTERPRISE_LLM_AVAILABLE = False


def initialize_session_state():
    """Initialize session state variables."""
    if "markdown_content" not in st.session_state:
        st.session_state.markdown_content = ""
    if "file_name" not in st.session_state:
        st.session_state.file_name = ""
    if "folder_processing_results" not in st.session_state:
        st.session_state.folder_processing_results = None


def check_enterprise_status():
    """Check and return enterprise LLM status."""
    if not ENTERPRISE_LLM_AVAILABLE:
        return False, "Enterprise LLM module not available"

    try:
        status = get_enterprise_llm_status()
        return status['configured'], status.get('message', '')
    except Exception as e:
        return False, f"Status check failed: {str(e)}"


def display_ai_status_sidebar():
    """Display AI enhancement status in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 AI Enhancement Status")

        enterprise_configured, message = check_enterprise_status()

        if enterprise_configured:
            st.success("✅ Enterprise LLM: Ready")
            st.caption("Using SageMaker endpoints")
            return True, None
        else:
            st.warning("⚠️ Enterprise LLM: Not configured")
            st.caption(f"Issue: {message}")

            # Claude fallback
            st.markdown("**Fallback Options:**")
            api_key_claude = st.text_input(
                "Claude API Key",
                type="password",
                help="Anthropic API key for Claude enhancement"
            )

            if api_key_claude:
                st.info("✅ Claude: Ready")
            else:
                st.info("📄 MarkItDown: Baseline only")

            return False, api_key_claude


def render_file_upload():
    """Enhanced file upload with proper enterprise routing."""
    st.header("📄 Upload Single File")

    uploaded_file = st.file_uploader(
        "Choose a file to convert",
        type=['pptx', 'ppt', 'docx', 'doc', 'pdf', 'xlsx', 'xls', 'html', 'htm', 'csv', 'json', 'xml'],
        help="Supported formats: PowerPoint, Word, PDF, Excel, HTML, CSV, JSON, XML"
    )

    if uploaded_file is not None:
        st.info(f"📁 **File:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔄 Convert to Markdown", type="primary"):
                enterprise_configured, claude_key = check_enterprise_status()

                with st.spinner("🚀 Processing..."):
                    try:
                        file_data = uploaded_file.read()

                        if enterprise_configured:
                            # Use enterprise LLM
                            st.info("🚀 Using Enterprise LLM...")
                            markdown_content, error = convert_file_to_markdown_enterprise(
                                file_data, uploaded_file.name, enhance=True
                            )
                        else:
                            # Use fallback converter
                            from src.converters.file_converter import convert_file_to_markdown
                            st.info("🔄 Using fallback converter...")
                            markdown_content, error = convert_file_to_markdown(
                                file_data, uploaded_file.name,
                                enhance=bool(claude_key), api_key=claude_key
                            )

                        if error:
                            st.error(f"❌ Error: {error}")
                        else:
                            st.session_state.markdown_content = markdown_content
                            st.session_state.file_name = uploaded_file.name

                            if enterprise_configured:
                                st.success("✅ Converted with Enterprise LLM!")
                            else:
                                st.success("✅ Converted successfully!")

                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")

        with col2:
            if st.button("ℹ️ File Info"):
                st.json({
                    "filename": uploaded_file.name,
                    "size_bytes": uploaded_file.size,
                    "size_mb": round(uploaded_file.size / 1024 / 1024, 2),
                    "type": uploaded_file.type
                })


def render_folder_processing():
    """Enhanced folder processing with proper enterprise routing."""
    st.header("📁 Process Folder")

    folder_path = st.text_input(
        "📂 Folder Path",
        placeholder="Enter the path to your folder containing files...",
        help="Path to folder containing documents to convert"
    )

    if folder_path:
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🚀 Process Folder", type="primary"):
                if not folder_path.strip() or not os.path.exists(folder_path):
                    st.error("Please enter a valid folder path")
                    return

                enterprise_configured, claude_key = check_enterprise_status()

                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    if enterprise_configured:
                        # Use enterprise folder processing
                        status_text.info("🚀 Using Enterprise LLM for folder processing...")

                        results = None
                        for progress_info in process_folder_enterprise(
                                folder_path, enhance=True
                        ):
                            if isinstance(progress_info, tuple) and len(progress_info) == 2:
                                progress, message = progress_info
                                progress_bar.progress(progress)
                                status_text.text(message)
                            else:
                                results = progress_info
                                break
                    else:
                        # Use fallback folder processing
                        from src.processors.folder_processor import process_folder
                        status_text.info("🔄 Using fallback folder processing...")

                        results = None
                        for progress_info in process_folder(
                                folder_path, enhance=bool(claude_key), api_key=claude_key
                        ):
                            if isinstance(progress_info, tuple) and len(progress_info) == 2:
                                progress, message = progress_info
                                progress_bar.progress(progress)
                                status_text.text(message)
                            else:
                                results = progress_info
                                break

                    if results:
                        success_count, error_count, errors = results

                        st.session_state.folder_processing_results = {
                            'success_count': success_count,
                            'error_count': error_count,
                            'errors': errors,
                            'total_files': success_count + error_count
                        }

                        if error_count == 0:
                            st.success(f"✅ Successfully processed {success_count} files!")
                        else:
                            st.warning(f"⚠️ Processed {success_count} files, {error_count} had errors")

                except Exception as e:
                    st.error(f"❌ Folder processing failed: {str(e)}")
                finally:
                    progress_bar.empty()
                    status_text.empty()

        with col2:
            if st.button("📊 Folder Statistics"):
                try:
                    from src.processors.folder_processor import get_folder_statistics
                    stats = get_folder_statistics(folder_path)

                    if "error" in stats:
                        st.error(stats["error"])
                    else:
                        st.json(stats)
                except Exception as e:
                    st.error(f"❌ Could not get folder statistics: {str(e)}")


def render_system_status():
    """Render system status tab."""
    st.header("📊 System Status")

    enterprise_configured, message = check_enterprise_status()

    # Enterprise LLM Status
    if enterprise_configured:
        st.success("✅ **Enterprise LLM**: Configured and Ready")
        st.caption("Using SageMaker endpoints with intelligent model routing")

        # Show configuration details
        with st.expander("🔍 Enterprise Configuration", expanded=False):
            try:
                status = get_enterprise_llm_status()
                st.json(status)
            except:
                st.error("Could not load detailed status")

    elif ENTERPRISE_LLM_AVAILABLE:
        st.warning("⚠️ **Enterprise LLM**: Available but not configured")
        st.caption(f"Issue: {message}")

        # Configuration help
        with st.expander("🔧 Configuration Help", expanded=True):
            st.markdown("""
            **Required Files:**
            - `JWT_token.txt` - Your authentication token
            - `model_url.txt` - SageMaker endpoint URL(s)

            **Example model_url.txt:**
            ```
            https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/your-model/invocations
            ```

            **Or for multiple models:**
            ```json
            {
                "metadata": "https://your-metadata-endpoint/invocations",
                "content": "https://your-content-endpoint/invocations", 
                "diagram": "https://your-diagram-endpoint/invocations"
            }
            ```
            """)
    else:
        st.error("❌ **Enterprise LLM**: Not Available")
        st.caption("Enterprise module not found")

    # Fallback systems
    st.subheader("🔄 Fallback Systems")

    try:
        from src.converters.claude_markdown_convertor import ClaudeMarkdownEnhancer
        st.info("✅ **Claude**: Available")
    except:
        st.error("❌ **Claude**: Not Available")

    st.info("✅ **MarkItDown**: Available (baseline)")

    # Show preferred method
    if enterprise_configured:
        st.success("🎯 **Current Method**: Enterprise LLM")
    else:
        st.warning("🎯 **Current Method**: Claude/MarkItDown Fallback")


def main():
    """Main application function."""
    # Set up page configuration
    setup_page_config()

    # Initialize session state
    initialize_session_state()

    # App header
    render_welcome_section()

    # Display AI status in sidebar
    display_ai_status_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗂️ Upload File", "🗃️ Process Folder", "📊 System Status", "📘 About"])

    with tab1:
        render_file_upload()

    with tab2:
        render_folder_processing()

    with tab3:
        render_system_status()

    with tab4:
        render_about_tab()

    # Display results sections
    display_enhanced_output_section()

    # Display diagram screenshot section
    try:
        from src.ui.diagram_screenshot import render_diagram_screenshot_section
        render_diagram_screenshot_section()
    except ImportError:
        pass

    # Display folder processing results
    display_folder_results()


if __name__ == "__main__":
    main()

