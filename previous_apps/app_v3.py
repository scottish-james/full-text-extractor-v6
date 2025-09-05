"""
Document to Markdown Converter - Simple Demo
Created by James Taylor
"""

import streamlit as st
import os

# Import your existing converters
try:
    from src.converters.file_converter import convert_file_to_markdown

    CONVERTER_AVAILABLE = True
except ImportError:
    CONVERTER_AVAILABLE = False


def setup_minimal_styling():
    """Clean, dark-mode friendly styling."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Global styling */
    .stApp {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }

    /* Hero section */
    .hero {
        text-align: center;
        padding: 3rem 0;
        margin-bottom: 3rem;
    }

    .hero h1 {
        font-size: 2.5rem;
        font-weight: 300;
        margin-bottom: 0.5rem;
        color: var(--text-color);
    }

    .hero p {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-bottom: 2rem;
    }

    .credit {
        font-size: 0.9rem;
        opacity: 0.6;
        margin-top: 1rem;
    }

    /* Card styling */
    .upload-card {
        background: var(--background-color);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
    }

    /* Toggle styling */
    .toggle-section {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
        padding: 1.5rem;
        background: var(--secondary-background-color);
        border-radius: 8px;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        height: 3rem;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        border: none;
        background: #4CAF50;
        color: white;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #45a049;
        transform: translateY(-1px);
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed var(--border-color);
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        transition: border-color 0.2s ease;
    }

    .stFileUploader:hover {
        border-color: #4CAF50;
    }

    /* Output section */
    .output-section {
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid var(--border-color);
    }

    /* Dark mode variables */
    [data-theme="dark"] {
        --text-color: #ffffff;
        --background-color: #1e1e1e;
        --secondary-background-color: #2d2d2d;
        --border-color: #404040;
    }

    /* Light mode variables */
    [data-theme="light"] {
        --text-color: #000000;
        --background-color: #ffffff;
        --secondary-background-color: #f8f9fa;
        --border-color: #e0e0e0;
    }

    /* Auto detect theme */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-color: #ffffff;
            --background-color: #1e1e1e;
            --secondary-background-color: #2d2d2d;
            --border-color: #404040;
        }
    }

    @media (prefers-color-scheme: light) {
        :root {
            --text-color: #000000;
            --background-color: #ffffff;
            --secondary-background-color: #f8f9fa;
            --border-color: #e0e0e0;
        }
    }

    /* Metrics styling */
    .metric-container {
        display: flex;
        gap: 2rem;
        justify-content: center;
        margin: 1rem 0;
    }

    .metric {
        text-align: center;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4CAF50;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.7;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session():
    """Initialize session state."""
    if "markdown_content" not in st.session_state:
        st.session_state.markdown_content = ""
    if "file_name" not in st.session_state:
        st.session_state.file_name = ""
    if "use_ai" not in st.session_state:
        st.session_state.use_ai = True


def render_hero():
    """Simple hero section."""
    st.markdown("""
    <div class="hero">
        <h1>DocFlow</h1>
        <p>Transform documents into clean markdown</p>
        <div class="credit">Created by James Taylor</div>
    </div>
    """, unsafe_allow_html=True)


def render_ai_toggle():
    """Simple AI toggle."""
    st.markdown('<div class="toggle-section">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        use_ai = st.toggle(
            "✨ AI Enhancement",
            value=st.session_state.use_ai,
            help="Use Claude AI to improve the markdown output"
        )
        st.session_state.use_ai = use_ai

        if use_ai:
            api_key = st.text_input(
                "Claude API Key",
                type="password",
                placeholder="sk-ant-...",
                help="Enter your Anthropic API key"
            )
        else:
            api_key = None

    st.markdown('</div>', unsafe_allow_html=True)

    return use_ai, api_key


def render_upload():
    """Simple file upload."""
    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose your file",
        type=['pptx', 'ppt', 'docx', 'doc', 'pdf', 'xlsx', 'xls', 'html', 'htm', 'csv', 'json', 'xml'],
        help="Drag and drop or click to browse",
        label_visibility="collapsed"
    )

    return uploaded_file


def convert_file(uploaded_file, use_ai, api_key):
    """Convert the uploaded file."""
    if not CONVERTER_AVAILABLE:
        st.error("❌ Converter not available - please check your installation")
        return

    with st.spinner("Converting your document..."):
        try:
            file_data = uploaded_file.getbuffer()

            # Convert file
            markdown_content, error = convert_file_to_markdown(
                file_data,
                uploaded_file.name,
                enhance=use_ai,
                api_key=api_key if use_ai else None
            )

            if error:
                st.error(f"Conversion failed: {error}")
            else:
                st.session_state.markdown_content = markdown_content
                st.session_state.file_name = uploaded_file.name

                if use_ai and api_key:
                    st.success("✨ Converted with AI enhancement!")
                else:
                    st.success("✅ Converted successfully!")

        except Exception as e:
            st.error(f"Error: {str(e)}")


def render_output():
    """Simple output display."""
    if not st.session_state.markdown_content:
        return

    st.markdown("---")
    st.markdown("### 📝 Results")

    content = st.session_state.markdown_content

    # Quick stats
    word_count = len(content.split())
    char_count = len(content)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Words", f"{word_count:,}")
    with col2:
        st.metric("Characters", f"{char_count:,}")
    with col3:
        filename = st.session_state.file_name.rsplit(".", 1)[0] + ".md"
        st.download_button(
            "📥 Download",
            data=content,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True
        )

    # Content preview
    st.text_area(
        "Markdown Content",
        value=content,
        height=300,
        help="Your converted markdown",
        label_visibility="collapsed"
    )


def main():
    """Main app function."""
    # Setup
    st.set_page_config(
        page_title="DocFlow - Simple Demo",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    setup_minimal_styling()
    initialize_session()

    # Render app
    render_hero()

    use_ai, api_key = render_ai_toggle()

    st.markdown("---")

    uploaded_file = render_upload()

    # Convert button
    if uploaded_file:
        st.markdown(f"**Selected:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")

        if st.button("🚀 Convert to Markdown", type="primary"):
            if use_ai and not api_key:
                st.error("Please enter your Claude API key to use AI enhancement")
            else:
                convert_file(uploaded_file, use_ai, api_key)

    # Show output
    render_output()


if __name__ == "__main__":
    main()