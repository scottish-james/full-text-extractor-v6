"""
Document to Markdown Converter - Enhanced with Smart Hyperlink Integration
Enterprise LLM Version
Created by James Taylor
"""

import streamlit as st
import os
import re
import pandas as pd
import tempfile
import base64
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Optional, Tuple
import logging
import requests

# Import your existing converters
try:
    from src.converters.file_converter import convert_file_to_markdown

    CONVERTER_AVAILABLE = True
except ImportError:
    CONVERTER_AVAILABLE = False

# Import DocVision components
try:
    from PIL import Image
    from pdf2image import convert_from_path

    DOCVISION_AVAILABLE = True
except ImportError:
    DOCVISION_AVAILABLE = False


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

    /* Output section */
    .output-section {
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid var(--border-color);
    }

    /* Status styling */
    .status-card {
        background: var(--secondary-background-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
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
    </style>
    """, unsafe_allow_html=True)


def initialize_session():
    """Initialize session state."""
    if "markdown_content" not in st.session_state:
        st.session_state.markdown_content = ""
    if "file_name" not in st.session_state:
        st.session_state.file_name = ""
    if "extracted_hyperlinks" not in st.session_state:
        st.session_state.extracted_hyperlinks = []
    if "xml_markdown" not in st.session_state:
        st.session_state.xml_markdown = ""
    if "vision_enhanced" not in st.session_state:
        st.session_state.vision_enhanced = False


def render_hero():
    """Simple hero section."""
    st.markdown("""
    <div class="hero">
        <h1>DocFlow Vision</h1>
        <p>AI-powered document conversion with smart hyperlink preservation</p>
        <div class="credit">Created by James Taylor - Enterprise Edition</div>
    </div>
    """, unsafe_allow_html=True)


def extract_hyperlinks_from_markdown(markdown_content):
    """
    Extract all hyperlinks from markdown content.
    Returns: list of dicts with Link Text, URL, and Slide Number
    """
    if not markdown_content:
        return []

    hyperlinks = []
    current_slide = 1

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
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.finditer(link_pattern, line)

        for match in matches:
            link_text = match.group(1)
            link_url = match.group(2)

            # Clean up the link text
            clean_text = re.sub(r'\*{1,3}', '', link_text).strip()

            # Skip image links and placeholders
            if clean_text.startswith('!') or link_url == "image":
                continue

            hyperlinks.append({
                'Link Text': clean_text,
                'URL': link_url,
                'Slide Number': current_slide
            })

    return hyperlinks


def create_enhanced_prompt_for_slide(slide_number: int, hyperlinks_df: pd.DataFrame) -> str:
    """
    Create an enhanced prompt for a specific slide if it has hyperlinks.

    Args:
        slide_number: The slide number being processed
        hyperlinks_df: DataFrame containing all hyperlinks

    Returns:
        Enhanced prompt string or None if no hyperlinks for this slide
    """
    # Filter for this slide's hyperlinks
    slide_links = hyperlinks_df[hyperlinks_df['Slide Number'] == slide_number]

    if slide_links.empty:
        return None

    # Build the hyperlink instruction
    link_instructions = []
    for _, row in slide_links.iterrows():
        link_instructions.append(f'[{row["Link Text"]}]({row["URL"]})')

    enhanced_instruction = f"""

IMPORTANT: This slide contains the following hyperlinks that must be preserved exactly:
{', '.join(link_instructions)}

Please ensure these hyperlinks are added appropriately in the markdown output where the link text appears.
"""

    return enhanced_instruction


class EnterpriseLLMClient:
    """Client for connecting to enterprise LLM endpoints."""

    def __init__(self, jwt_token: str, model_url: str):
        """
        Initialise enterprise LLM client.

        Args:
            jwt_token: JWT authentication token
            model_url: Enterprise model endpoint URL
        """
        self.jwt_token = jwt_token
        self.model_url = model_url
        self.headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger(__name__)

    def test_connection(self) -> bool:
        """Test connectivity to the enterprise endpoint."""
        try:
            response = requests.head(self.model_url, timeout=10)
            return response.status_code < 400
        except Exception as e:
            st.error(f"Enterprise endpoint test failed: {e}")
            return False

    def extract_text_from_image(self, image: Image.Image, prompt: str) -> Optional[str]:
        """
        Extract text from image using enterprise LLM.

        Args:
            image: PIL Image object
            prompt: System prompt for processing

        Returns:
            Extracted text or None if failed
        """
        try:
            # Convert image to base64
            buffer = BytesIO()
            image.save(buffer, format='PNG', optimize=True)
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Prepare payload for enterprise LLM
            payload = {
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Convert this image to Markdown:\n[Image attached]"}
                ],
                "max_tokens": 4000,
                "temperature": 0.1,
                "image_data": image_b64  # Custom field for enterprise endpoint
            }

            # Make request
            response = requests.post(
                self.model_url,
                headers=self.headers,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()

                # Try different response formats
                if "choices" in result and result["choices"]:
                    return result["choices"][0]["message"]["content"]
                elif "generated_text" in result:
                    return result["generated_text"]
                elif "content" in result:
                    return result["content"]
                else:
                    return str(result)
            else:
                st.error(f"Enterprise LLM error {response.status_code}: {response.text}")
                return None

        except Exception as e:
            st.error(f"Enterprise LLM request failed: {e}")
            return None


class EnhancedDocVision:
    """Extended DocVision class with hyperlink-aware processing using Enterprise LLM."""

    def __init__(self, jwt_token: str, model_url: str):
        """Initialize with Enterprise LLM credentials."""
        if not jwt_token or not model_url:
            raise ValueError("Enterprise LLM JWT token and model URL are required for vision processing")

        self.client = EnterpriseLLMClient(jwt_token, model_url)
        self.base_prompt = """You are a PowerPoint slide to Markdown converter. You receive an image of a PowerPoint slide and must convert all 
visible text and structure into clean, professional markdown.

Your job:
1. Extract ALL text content from the slide
2. Identify the main title and format as # header (only ONE per slide)
3. Identify subtitles or section headers and format as ### headers
4. Convert bullet points to proper markdown lists with correct indentation (use 2 spaces for nested bullets)
5. Preserve table structures using markdown table syntax
6. Extract and format any numbered lists
7. Maintain slide hierarchy and structure
8. Include any visible links, captions, or annotations
9. Format code blocks or technical content appropriately
10. If there are diagrams then create them in mermaid code format that can be used in a .md file. Diagrams will have 
text boxes and arrows. If you only see text boxes but no lines or arrows this is NOT a diagram. THEREFORE IT MUST 
HAVE LINES BETWEEN TEXT BOX. If not just extract text. 

Key Rules:
- Extract ALL visible text - don't miss anything
- Each slide should have only ONE main # heading
- Use ### for subheadings (not ##)
- Use proper markdown syntax throughout
- Maintain the original slide's logical structure
- If text is unclear, make your best reasonable interpretation
- Don't add content that isn't visible in the image
- Format tables properly with | separators
- Preserve bullet point hierarchies with proper indentation

Output clean, readable markdown that captures everything visible on the slide.

Finally - all this output is going to an .md file so you DO NOT NEED to put ```markdown ARE WE CLEAR"""

    def process_with_hyperlinks(self, file_path: Path, hyperlinks_df: pd.DataFrame) -> str:
        """
        Process a document with hyperlink awareness.

        Args:
            file_path: Path to the document
            hyperlinks_df: DataFrame containing hyperlink information

        Returns:
            Markdown content with correct hyperlinks
        """
        # Convert to images first
        if file_path.suffix.lower() == '.pdf':
            images = self._pdf_to_images(file_path)
        elif file_path.suffix.lower() in ['.pptx', '.ppt']:
            images = self._powerpoint_to_images(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Process each slide/page
        markdown_sections = []

        for i, image in enumerate(images, 1):
            # Check if this slide has hyperlinks
            enhanced_instruction = create_enhanced_prompt_for_slide(i, hyperlinks_df)

            # Build the prompt
            if enhanced_instruction:
                prompt = self.base_prompt + enhanced_instruction
                st.info(
                    f"📎 Slide {i}: Processing with {len(hyperlinks_df[hyperlinks_df['Slide Number'] == i])} hyperlinks")
            else:
                prompt = self.base_prompt
                st.info(f"📄 Slide {i}: Standard processing")

            # Process the image
            markdown_text = self._extract_text_from_image(image, prompt)

            if markdown_text:
                markdown_sections.append(f"<!-- Slide {i} -->\n\n{markdown_text}")
            else:
                markdown_sections.append(f"<!-- Slide {i} -->\n\n*[Could not extract text from this slide]*")

        # Combine all sections
        header = f"# {file_path.name}\n\n*AI-enhanced conversion with preserved hyperlinks (Enterprise LLM)*\n\n"
        return header + "\n\n---\n\n".join(markdown_sections)

    def _pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF to images."""
        return convert_from_path(str(pdf_path), dpi=250, fmt='PNG')

    def _powerpoint_to_images(self, ppt_path: Path) -> List[Image.Image]:
        """Convert PowerPoint to images."""
        import subprocess
        import shutil

        # Find LibreOffice
        libreoffice_path = None
        for cmd in ["/Applications/LibreOffice.app/Contents/MacOS/soffice", "soffice", "libreoffice"]:
            if cmd.startswith("/"):
                if Path(cmd).exists():
                    libreoffice_path = cmd
                    break
            else:
                libreoffice_path = shutil.which(cmd)
                if libreoffice_path:
                    break

        if not libreoffice_path:
            st.error("LibreOffice not found. Please install it first.")
            st.info("Install with: brew install --cask libreoffice")
            return []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Convert PowerPoint to PDF using LibreOffice
                st.info("Converting PowerPoint to PDF...")
                cmd = [
                    libreoffice_path,
                    "--headless",
                    "--convert-to", "pdf",
                    "--outdir", str(temp_path),
                    str(ppt_path)
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode != 0:
                    st.error(f"LibreOffice conversion failed: {result.stderr}")
                    return []

                # Find generated PDF
                pdf_files = list(temp_path.glob("*.pdf"))
                if not pdf_files:
                    st.error("PDF not generated from PowerPoint")
                    return []

                pdf_path = pdf_files[0]

                # Convert PDF to images
                st.info("Converting PDF to images...")
                images = convert_from_path(str(pdf_path), dpi=250, fmt='PNG')
                return images

            except subprocess.TimeoutExpired:
                st.error("Conversion timeout - file may be too large")
                return []
            except Exception as e:
                st.error(f"Conversion error: {str(e)}")
                return []

    def _extract_text_from_image(self, image: Image.Image, prompt: str) -> str:
        """Extract text from image using Enterprise LLM."""
        try:
            # Resize if needed
            if max(image.size) > 2048:
                ratio = 2048 / max(image.size)
                new_size = tuple(int(d * ratio) for d in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to RGB if needed
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # Use Enterprise LLM client
            return self.client.extract_text_from_image(image, prompt)

        except Exception as e:
            st.error(f"Vision processing error: {e}")
            return None


def load_enterprise_config():
    """Load enterprise LLM configuration from files or environment."""
    jwt_token = None
    model_url = None

    # Try to load from files first
    try:
        if os.path.exists("JWT_token.txt"):
            with open("JWT_token.txt", "r") as f:
                jwt_token = f.read().strip()

        if os.path.exists("model_url.txt"):
            with open("model_url.txt", "r") as f:
                model_url = f.read().strip()
    except Exception as e:
        st.warning(f"Could not read config files: {e}")

    # Fall back to environment variables
    if not jwt_token:
        jwt_token = os.getenv("ENTERPRISE_JWT_TOKEN")
    if not model_url:
        model_url = os.getenv("ENTERPRISE_MODEL_URL")

    return jwt_token, model_url


def convert_file_enhanced(uploaded_file, use_vision: bool, jwt_token: str, model_url: str):
    """
    Enhanced conversion with smart hyperlink handling using Enterprise LLM.

    1. First run XML extractor to get ground truth
    2. Extract hyperlinks from XML output
    3. If using vision, process with hyperlink awareness using Enterprise LLM
    """
    if not CONVERTER_AVAILABLE:
        st.error("❌ Converter not available - please check your installation")
        return

    with st.spinner("🔍 Extracting document structure..."):
        try:
            file_data = uploaded_file.getbuffer()

            # Step 1: Run XML extractor to get ground truth
            st.info("📊 Step 1: Extracting document structure and hyperlinks...")
            xml_markdown, error = convert_file_to_markdown(
                file_data,
                uploaded_file.name,
                enhance=False,  # Don't use AI enhancement for XML extraction
                api_key=None
            )

            if error:
                st.error(f"XML extraction failed: {error}")
                return

            st.session_state.xml_markdown = xml_markdown

            # Step 2: Extract hyperlinks from XML output
            st.info("🔗 Step 2: Building hyperlink table...")
            hyperlinks = extract_hyperlinks_from_markdown(xml_markdown)

            if hyperlinks:
                st.session_state.extracted_hyperlinks = hyperlinks
                hyperlinks_df = pd.DataFrame(hyperlinks)
                st.success(
                    f"✅ Found {len(hyperlinks)} hyperlinks across {hyperlinks_df['Slide Number'].nunique()} slides")

                # Show preview of hyperlinks
                with st.expander("Preview Hyperlink Table"):
                    st.dataframe(hyperlinks_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No hyperlinks found in document")
                hyperlinks_df = pd.DataFrame()
                st.session_state.extracted_hyperlinks = []

            # Step 3: Process with vision if requested
            if use_vision and jwt_token and model_url:
                if not DOCVISION_AVAILABLE:
                    st.error("Vision dependencies not installed. Please install: pdf2image, pillow")
                    return

                st.info("🤖 Step 3: Processing with Enterprise LLM Vision (hyperlink-aware)...")

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(file_data)
                    tmp_path = Path(tmp_file.name)

                try:
                    # Create enhanced DocVision processor with Enterprise LLM
                    processor = EnhancedDocVision(jwt_token, model_url)

                    # Test connection first
                    if not processor.client.test_connection():
                        st.warning("Enterprise endpoint may be unreachable, but proceeding with conversion...")

                    # Process with hyperlink awareness
                    vision_markdown = processor.process_with_hyperlinks(tmp_path, hyperlinks_df)

                    st.session_state.markdown_content = vision_markdown
                    st.session_state.vision_enhanced = True
                    st.success("✨ Enterprise LLM Vision processing complete with hyperlink preservation!")

                finally:
                    # Clean up temp file
                    if tmp_path.exists():
                        tmp_path.unlink()
            else:
                # Use XML extraction result
                st.session_state.markdown_content = xml_markdown
                st.session_state.vision_enhanced = False
                st.success("✅ Converted successfully using XML extraction!")

            st.session_state.file_name = uploaded_file.name

        except Exception as e:
            st.error(f"Error during conversion: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def render_upload():
    """File upload section."""
    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose your file",
        type=['pptx', 'ppt', 'pdf', 'docx', 'doc'],
        help="Upload a PowerPoint or PDF file",
        label_visibility="collapsed"
    )

    return uploaded_file


def render_output():
    """Output display with hyperlink extraction."""
    if not st.session_state.markdown_content:
        return

    st.markdown("---")
    st.markdown("### 📝 Results")

    # Show processing method
    if st.session_state.vision_enhanced:
        st.info("🏢 Enhanced with Enterprise LLM Vision + Hyperlink Preservation")
    else:
        st.info("📊 Processed with XML Extraction")

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
            "📥 Download Markdown",
            data=content,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True
        )

    # Content preview
    st.text_area(
        "Markdown Content",
        value=content,
        height=400,
        label_visibility="collapsed"
    )

    # Hyperlink section
    if st.session_state.extracted_hyperlinks:
        st.markdown("---")
        st.markdown("### 🔗 Extracted Hyperlinks")

        df = pd.DataFrame(st.session_state.extracted_hyperlinks)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Links", len(df))
        with col2:
            st.metric("Unique URLs", df['URL'].nunique())
        with col3:
            csv = df.to_csv(index=False)
            base_filename = st.session_state.file_name.rsplit(".", 1)[0]
            st.download_button(
                "💾 Download CSV",
                data=csv,
                file_name=f"{base_filename}_hyperlinks.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Link Text": st.column_config.TextColumn("Link Text", width="medium"),
                "URL": st.column_config.LinkColumn("URL", width="large"),
                "Slide Number": st.column_config.NumberColumn("Slide", width="small", format="%d")
            }
        )


def main():
    """Main application."""
    st.set_page_config(
        page_title="DocFlow Vision - Enterprise",
        page_icon="🔬",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    setup_minimal_styling()
    initialize_session()

    render_hero()

    st.markdown("---")

    # Configuration section
    st.markdown("### ⚙️ Enterprise Configuration")

    # Load existing configuration
    existing_jwt, existing_url = load_enterprise_config()

    col1, col2 = st.columns(2)

    with col1:
        use_vision = st.checkbox(
            "🤖 Use Enterprise Vision Processing",
            value=True,
            help="Process document images with Enterprise LLM Vision API for better formatting"
        )

    with col2:
        if existing_jwt:
            masked_token = f"{existing_jwt[:10]}..." if len(existing_jwt) > 10 else "***"
            st.info(f"✅ JWT Token loaded: {masked_token}")
        if existing_url:
            st.info(f"✅ Model URL loaded: {existing_url[:50]}...")

    if use_vision:
        st.markdown("#### Enterprise LLM Credentials")

        # Allow override of loaded config
        jwt_token = st.text_input(
            "JWT Token",
            type="password",
            placeholder="Enter JWT token or leave empty to use config file",
            help="JWT authentication token for Enterprise LLM"
        ) or existing_jwt

        model_url = st.text_input(
            "Model Endpoint URL",
            placeholder="Enter model URL or leave empty to use config file",
            help="Enterprise LLM endpoint URL"
        ) or existing_url

        if not jwt_token or not model_url:
            st.warning(
                "⚠️ Please provide Enterprise LLM credentials or ensure JWT_token.txt and model_url.txt files exist")
            st.info("""
            **Configuration Options:**
            1. Enter credentials above, or
            2. Create these files in the same directory:
               - `JWT_token.txt` (containing your JWT token)
               - `model_url.txt` (containing the model endpoint URL)
            3. Set environment variables:
               - `ENTERPRISE_JWT_TOKEN`
               - `ENTERPRISE_MODEL_URL`
            """)
    else:
        jwt_token = None
        model_url = None

    st.markdown("---")

    # File upload
    uploaded_file = render_upload()

    if uploaded_file:
        st.markdown(f"**Selected:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")

        if st.button("🚀 Convert to Markdown", type="primary"):
            if use_vision and (not jwt_token or not model_url):
                st.error("Please provide Enterprise LLM credentials for vision processing")
            else:
                convert_file_enhanced(uploaded_file, use_vision, jwt_token, model_url)

    # Show output
    render_output()


if __name__ == "__main__":
    main()

