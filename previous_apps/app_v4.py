"""
Document to Markdown Converter - Simple Demo with Hyperlink Extraction
Created by James Taylor
"""

import streamlit as st
import os
import re
import pandas as pd
from io import StringIO

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

    /* Hyperlink extraction section */
    .hyperlink-section {
        background: var(--secondary-background-color);
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 2rem;
    }

    /* Table styling for hyperlinks */
    .dataframe tbody tr:hover {
        background-color: var(--secondary-background-color);
    }

    /* Link styling in table */
    .dataframe a {
        color: #4CAF50;
        text-decoration: none;
    }

    .dataframe a:hover {
        text-decoration: underline;
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
    if "extracted_hyperlinks" not in st.session_state:
        st.session_state.extracted_hyperlinks = []


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


def extract_hyperlinks_from_markdown(markdown_content):
    """
    Extract all hyperlinks from markdown content.

    Patterns to match:
    - [text](url) - Standard markdown links
    - [**text**](url) - Bold text links
    - [*text*](url) - Italic text links
    - [***text***](url) - Bold+italic text links
    - Nested formatting patterns

    Returns:
        list: List of dicts with text, url, and slide number
    """
    if not markdown_content:
        return []

    hyperlinks = []
    current_slide = 1

    # Split content into lines for slide tracking
    lines = markdown_content.split('\n')

    for line in lines:
        # Track slide numbers from HTML comments or slide markers
        slide_comment_match = re.search(r'<!--\s*Slide\s*(\d+)\s*-->', line)
        if slide_comment_match:
            current_slide = int(slide_comment_match.group(1))
            continue

        # Also check for "Slide X" in headers
        slide_header_match = re.search(r'^#+\s*Slide\s*(\d+)', line)
        if slide_header_match:
            current_slide = int(slide_header_match.group(1))

        # Find all markdown links in the line
        # This regex captures links with various text formatting
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.finditer(link_pattern, line)

        for match in matches:
            link_text = match.group(1)
            link_url = match.group(2)

            # Clean up the link text (remove markdown formatting)
            clean_text = link_text
            # Remove bold markers
            clean_text = re.sub(r'\*{1,3}', '', clean_text)
            # Remove any remaining markdown formatting
            clean_text = clean_text.strip()

            # Skip if it's an image link
            if clean_text.startswith('!'):
                continue

            # Skip if URL is just "image" (placeholder for images)
            if link_url == "image":
                continue

            hyperlinks.append({
                'Link Text': clean_text,
                'URL': link_url,
                'Slide Number': current_slide
            })

    return hyperlinks


def render_hyperlink_extractor():
    """
    Render the hyperlink extraction section with CSV download.
    """
    if not st.session_state.markdown_content:
        return

    st.markdown("---")
    st.markdown("### 🔗 Hyperlink Extraction")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("📊 Extract Hyperlinks", type="secondary"):
            # Extract hyperlinks from markdown
            hyperlinks = extract_hyperlinks_from_markdown(st.session_state.markdown_content)

            if hyperlinks:
                # Store in session state for persistence
                st.session_state.extracted_hyperlinks = hyperlinks
                st.success(f"✅ Found {len(hyperlinks)} hyperlinks!")
            else:
                st.warning("No hyperlinks found in the document")
                st.session_state.extracted_hyperlinks = []

    # Display results if available
    if st.session_state.extracted_hyperlinks:
        with col2:
            # Create DataFrame
            df = pd.DataFrame(st.session_state.extracted_hyperlinks)

            # Generate CSV
            csv = df.to_csv(index=False)

            # Create filename
            base_filename = st.session_state.file_name.rsplit(".", 1)[0] if st.session_state.file_name else "document"
            csv_filename = f"{base_filename}_hyperlinks.csv"

            # Download button
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=csv_filename,
                mime="text/csv",
                use_container_width=True
            )

        # Display preview table
        st.markdown("#### Preview")

        # Show stats
        unique_urls = df['URL'].nunique()
        slides_with_links = df['Slide Number'].nunique()

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Total Links", len(st.session_state.extracted_hyperlinks))
        with metric_col2:
            st.metric("Unique URLs", unique_urls)
        with metric_col3:
            st.metric("Slides with Links", slides_with_links)

        # Display the DataFrame with better formatting
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Link Text": st.column_config.TextColumn(
                    "Link Text",
                    width="medium",
                ),
                "URL": st.column_config.LinkColumn(
                    "URL",
                    width="large",
                ),
                "Slide Number": st.column_config.NumberColumn(
                    "Slide",
                    width="small",
                    format="%d"
                )
            }
        )

        # Optional: Group by slide
        with st.expander("📑 View by Slide"):
            for slide_num in sorted(df['Slide Number'].unique()):
                st.markdown(f"**Slide {slide_num}**")
                slide_links = df[df['Slide Number'] == slide_num][['Link Text', 'URL']]
                for _, row in slide_links.iterrows():
                    st.markdown(f"- [{row['Link Text']}]({row['URL']})")


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

    # Add hyperlink extraction section
    render_hyperlink_extractor()


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