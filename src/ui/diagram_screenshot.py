"""
Complete Fixed Screenshot UI with OpenAI Integration and Persistence
Replace your entire src/ui/diagram_screenshot.py with this content
"""

import streamlit as st
import re
import tempfile
import os
import sys
import base64
from io import StringIO
from contextlib import redirect_stdout
from typing import List, Dict, Optional


def render_diagram_screenshot_section():
    """
    ENHANCED VERSION - Includes OpenAI integration and persistent results
    """
    st.markdown("---")
    st.subheader("📸 Diagram Screenshot Generator")

    # Check if we have OpenAI API key from sidebar
    if 'openai_api_key' not in st.session_state or not st.session_state.get('openai_api_key'):
        st.info("💡 **Tip:** Add your OpenAI API key in the sidebar to enable AI diagram analysis")

    # FIXED: Always show existing screenshot results first
    if 'screenshot_results' in st.session_state and st.session_state.screenshot_results:
        st.subheader("📊 Current Screenshot Results")
        _display_all_screenshot_results()

        # Add button to clear results
        if st.button("🗑️ Clear Screenshots", help="Remove current screenshots to generate new ones"):
            _clear_screenshot_results()
            st.rerun()

        st.markdown("---")

    # Step 1: Check session state
    st.markdown("**Session State Check**")
    has_content = 'markdown_content' in st.session_state and st.session_state.markdown_content
    has_filename = 'file_name' in st.session_state and st.session_state.file_name
    has_file_data = 'uploaded_file_data' in st.session_state

    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.write("✅ Content" if has_content else "❌ Content")
    with status_col2:
        st.write("✅ Filename" if has_filename else "❌ Filename")
    with status_col3:
        st.write("✅ File Data" if has_file_data else "❌ File Data")

    if not (has_content and has_filename and has_file_data):
        st.warning("⚠️ Upload and convert a PowerPoint file first to enable screenshots")
        return

    # Step 2: Check file type
    filename = st.session_state.file_name
    file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
    is_powerpoint = file_ext in ['pptx', 'ppt']

    if not is_powerpoint:
        st.error("❌ Screenshots only work with PowerPoint files (.pptx, .ppt)")
        return

    st.success(f"✅ PowerPoint file ready: {filename}")

    # Step 3: Parse v19 analysis for high-probability slides
    markdown_content = st.session_state.markdown_content
    high_prob_slides = _parse_v19_diagram_analysis(markdown_content)

    if not high_prob_slides:
        st.warning("⚠️ No slides with >80% diagram probability found")

        # Show manual slide selection option
        with st.expander("🔧 Manual Slide Selection", expanded=False):
            st.markdown("**Select slides manually if automatic detection didn't work:**")

            # Get total slide count from markdown
            total_slides = _count_total_slides(markdown_content)
            if total_slides > 0:
                st.write(f"Total slides in presentation: {total_slides}")

                # Manual slide number input
                manual_slides = st.text_input(
                    "Enter slide numbers (comma-separated):",
                    placeholder="e.g., 1, 3, 5",
                    help="Enter the slide numbers you want to screenshot"
                )

                if manual_slides and st.button("📸 Screenshot Selected Slides", type="secondary"):
                    try:
                        slide_numbers = [int(x.strip()) for x in manual_slides.split(',')]
                        slide_numbers = [s for s in slide_numbers if 1 <= s <= total_slides]
                        if slide_numbers:
                            _execute_improved_screenshots(slide_numbers)
                        else:
                            st.error("❌ Invalid slide numbers")
                    except ValueError:
                        st.error("❌ Please enter valid numbers separated by commas")
            else:
                st.write("Could not determine total slide count from markdown")
        return

    # Show detected high-probability slides
    st.success(f"🎯 Found {len(high_prob_slides)} slides with >80% diagram probability:")

    for slide in high_prob_slides:
        st.write(f"• **Slide {slide['slide_number']}**: {slide['probability']}% probability")

    # Main screenshot button
    slide_numbers = [s['slide_number'] for s in high_prob_slides]
    slide_text = ", ".join(map(str, slide_numbers))

    if st.button(f"📸 Screenshot Diagram Slides: {slide_text}", type="primary"):
        _execute_improved_screenshots(slide_numbers)


def _parse_v19_diagram_analysis(markdown_content: str) -> List[Dict]:
    """Parse v19 analysis for slides >80%."""
    high_prob_slides = []

    # Find analysis section
    analysis_pattern = r'# DIAGRAM ANALYSIS.*?\*\*Slides with potential diagrams:\*\*(.*?)(?=\n#|\Z)'
    analysis_match = re.search(analysis_pattern, markdown_content, re.DOTALL | re.IGNORECASE)

    if not analysis_match:
        return high_prob_slides

    analysis_section = analysis_match.group(1)

    # Parse slide lines
    slide_pattern = r'- \*\*Slide (\d+)\*\*: (\d+)% probability'

    for match in re.finditer(slide_pattern, analysis_section):
        slide_num = int(match.group(1))
        probability = int(match.group(2))

        if probability > 80:
            high_prob_slides.append({
                'slide_number': slide_num,
                'probability': probability
            })

    return sorted(high_prob_slides, key=lambda x: x['probability'], reverse=True)


def _count_total_slides(markdown_content: str) -> int:
    """Count total slides from slide markers in markdown."""
    slide_markers = re.findall(r'<!-- Slide (\d+) -->', markdown_content)
    if slide_markers:
        return max(int(marker) for marker in slide_markers)
    return 0


def _execute_improved_screenshots(slide_numbers: List[int]):
    """Execute screenshots using the new improved method with fallback."""
    st.markdown("---")
    st.subheader("📸 Screenshot Generation Process")

    try:
        # Setup
        filename = st.session_state.file_name
        file_ext = filename.split('.')[-1].lower() if '.' in filename else 'pptx'
        file_data = st.session_state.uploaded_file_data

        st.write(f"📁 **File:** {filename}")
        st.write(f"📊 **Size:** {len(file_data) / (1024 * 1024):.1f} MB")
        st.write(f"🎯 **Target slides:** {slide_numbers}")

        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as temp_pptx:
            temp_pptx.write(file_data)
            temp_pptx_path = temp_pptx.name

        # Test LibreOffice availability
        from src.processors.diagram_screenshot_processor import test_diagram_screenshot_capability, \
            install_poppler_instructions
        available, status = test_diagram_screenshot_capability()
        st.write(f"🔧 **LibreOffice Status:** {status}")

        if not available:
            st.error("❌ LibreOffice not available - required for screenshot generation")
            return

        # Show poppler installation tip if not available
        if "without Poppler" in status:
            with st.expander("💡 Improve Screenshot Quality"):
                st.markdown("**Install poppler-utils for better screenshot quality:**")
                instructions = install_poppler_instructions()
                st.code(instructions)
                st.info(
                    "Poppler-utils provides more reliable PDF page extraction. The tool will work without it, but with poppler you'll get better results.")

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Screenshot with improved method
        with tempfile.TemporaryDirectory() as output_dir:
            from src.processors.diagram_screenshot_processor import DiagramScreenshotProcessor
            processor = DiagramScreenshotProcessor()

            status_text.text("📸 Generating screenshots...")
            progress_bar.progress(25)

            # Capture debug output for display
            debug_output = StringIO()

            with st.spinner("📸 Generating screenshots..."):
                # Redirect print statements to capture debug info
                with redirect_stdout(debug_output):
                    # Use the correct method name
                    results = processor.screenshot_slides_with_all_methods(
                        temp_pptx_path,
                        slide_numbers,
                        output_dir,
                        filename.rsplit('.', 1)[0],
                        debug_mode=True  # Show debug info
                    )

            # Show debug output
            debug_text = debug_output.getvalue()
            if debug_text:
                with st.expander("🔍 LibreOffice Debug Output"):
                    st.code(debug_text)

            progress_bar.progress(75)
            status_text.text("Processing results...")

            # FIXED: Store screenshot results in session state
            if results:
                # Store results with image data in session state
                screenshot_results = {}
                for slide_num, file_path in results.items():
                    if os.path.exists(file_path):
                        # Read and store the image data
                        with open(file_path, "rb") as f:
                            image_data = f.read()

                        screenshot_results[slide_num] = {
                            'file_path': file_path,
                            'image_data': image_data,
                            'filename': os.path.basename(file_path),
                            'file_size': os.path.getsize(file_path) / 1024
                        }

                # Store in session state
                st.session_state.screenshot_results = screenshot_results
                st.session_state.screenshot_slide_numbers = slide_numbers

            # Clear progress indicators
            progress_bar.progress(100)
            status_text.text("✅ Screenshot generation complete!")

            # Show results
            st.markdown("---")
            st.subheader("📊 Screenshot Results")

            if results:
                success_count = len(results)
                total_count = len(slide_numbers)

                if success_count == total_count:
                    st.success(f"🎉 Successfully generated {success_count} screenshots!")
                else:
                    st.warning(f"⚠️ Generated {success_count} of {total_count} requested screenshots")

                # Display each screenshot using stored results
                _display_all_screenshot_results()

            else:
                st.error("❌ No screenshots were generated")
                st.info("💡 **Troubleshooting:**")
                st.markdown("""
                - Make sure LibreOffice is properly installed
                - Try with different slides
                - Check if the PowerPoint file is valid
                - Some slides might not be exportable due to complex formatting
                """)

        # Cleanup
        os.unlink(temp_pptx_path)
        progress_bar.empty()
        status_text.empty()

    except Exception as e:
        st.error(f"❌ **Error during screenshot generation:** {str(e)}")

        with st.expander("🐛 Debug Information"):
            import traceback
            st.code(traceback.format_exc())


def _display_all_screenshot_results():
    """Display all screenshot results from session state."""
    if 'screenshot_results' not in st.session_state:
        return

    screenshot_results = st.session_state.screenshot_results
    slide_numbers = st.session_state.get('screenshot_slide_numbers', [])

    # Display each screenshot
    for slide_num in slide_numbers:
        if slide_num in screenshot_results:
            _display_screenshot_result_from_session(slide_num)
        else:
            st.error(f"❌ **Slide {slide_num}**: Screenshot generation failed")


def _display_screenshot_result_from_session(slide_num: int):
    """Display a single screenshot result from session state data."""
    if 'screenshot_results' not in st.session_state or slide_num not in st.session_state.screenshot_results:
        st.error(f"❌ Slide {slide_num}: Screenshot data not found")
        return

    result_data = st.session_state.screenshot_results[slide_num]

    st.markdown(f"### ✅ Slide {slide_num}")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Display thumbnail using stored image data
        try:
            st.image(result_data['image_data'], caption=f"Slide {slide_num}")  # Full resolution
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")

    with col2:
        # File information
        st.write(f"**📄 Filename:** {result_data['filename']}")
        st.write(f"**📊 Size:** {result_data['file_size']:.1f} KB")

        # Button row with download and analyze
        try:
            col_download, col_analyze = st.columns(2)

            with col_download:
                st.download_button(
                    f"📥 Download",
                    data=result_data['image_data'],
                    file_name=result_data['filename'],
                    mime="image/png",
                    key=f"download_slide_{slide_num}",
                    use_container_width=True
                )

            with col_analyze:
                # Check if OpenAI key is available
                has_openai_key = 'openai_api_key' in st.session_state and st.session_state.openai_api_key

                if has_openai_key:
                    # OpenAI analysis button
                    if st.button(
                            f"🤖 Analyze with AI",
                            key=f"analyze_slide_{slide_num}",
                            use_container_width=True,
                            help="Use OpenAI to convert this diagram to Mermaid code"
                    ):
                        _analyze_screenshot_with_openai(slide_num, result_data['file_path'], result_data['image_data'])
                else:
                    st.button(
                        f"🔑 Need OpenAI Key",
                        disabled=True,
                        use_container_width=True,
                        help="Add OpenAI API key in sidebar to enable AI analysis"
                    )

        except Exception as e:
            st.error(f"Error creating buttons: {str(e)}")

    # FIXED: Show any AI analysis results for this slide
    _display_ai_analysis_results(slide_num)

    st.markdown("---")


def _display_ai_analysis_results(slide_num: int):
    """Display AI analysis results if they exist for this slide."""
    ai_results_key = f"ai_analysis_{slide_num}"

    if ai_results_key in st.session_state:
        analysis_data = st.session_state[ai_results_key]

        st.markdown(f"### 🤖 AI Analysis for Slide {slide_num}")
        st.success("✅ Analysis complete!")

        # Display the generated Mermaid code
        st.subheader("🎯 Generated Mermaid Diagram")
        st.code(analysis_data['mermaid_code'], language="mermaid")

        # Show rendered preview
        try:
            st.subheader("📊 Diagram Preview")
            st.write("```mermaid")
            st.write(analysis_data['mermaid_code'])
            st.write("```")
        except:
            st.info("💡 Copy the code above and paste it into a Mermaid renderer to see the preview")

        # Download button for the Mermaid code
        st.download_button(
            "📥 Download Mermaid Code",
            data=analysis_data['mermaid_code'],
            file_name=f"slide_{slide_num}_diagram.mmd",
            mime="text/plain",
            key=f"download_mermaid_{slide_num}_persistent",
            help="Download the Mermaid diagram code"
        )

        # Option to add to markdown
        if st.button(f"➕ Add to Markdown", key=f"add_to_md_{slide_num}_persistent"):
            _add_mermaid_to_session_markdown(slide_num, analysis_data['mermaid_code'])


def _analyze_screenshot_with_openai(slide_num: int, file_path: str, image_data: bytes):
    """Analyze screenshot with OpenAI and store results in session state."""

    # Get OpenAI API key from session state
    openai_api_key = st.session_state.get('openai_api_key')

    if not openai_api_key:
        st.error("❌ OpenAI API key not found. Please add it in the sidebar.")
        return

    with st.spinner("🔍 Analyzing diagram with OpenAI..."):
        try:
            # Call OpenAI API
            mermaid_code = _call_openai_vision_api(image_data, openai_api_key)

            if mermaid_code:
                # FIXED: Store analysis results in session state
                ai_results_key = f"ai_analysis_{slide_num}"
                st.session_state[ai_results_key] = {
                    'mermaid_code': mermaid_code,
                    'analyzed': True
                }

                # Force a rerun to show the results
                st.rerun()
            else:
                st.error("❌ Failed to generate Mermaid diagram")

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")


def _call_openai_vision_api(image_data: bytes, api_key: str) -> Optional[str]:
    """Call OpenAI Vision API to analyze the diagram and generate Mermaid code."""
    try:
        import openai

        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)

        # Convert image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Prepare the prompt
        system_prompt = """You are an expert at analyzing diagrams and converting them to Mermaid diagram code."""

        user_prompt = "Please analyze this diagram and convert it to Mermaid code following the guidelines."

        # Make API call
        response = client.chat.completions.create(
            model="gpt-4o",  # Using the best vision model
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=16384,
            temperature=0.1  # Low temperature for consistent output
        )

        mermaid_code = response.choices[0].message.content.strip()

        # # Clean up the response (remove any markdown formatting that might slip through)
        # if mermaid_code.startswith("```"):
        #     lines = mermaid_code.split('\n')
        #     # Remove first and last lines if they're markdown code blocks
        #     if lines[0].startswith("```"):
        #         lines = lines[1:]
        #     if lines and lines[-1].strip() == "```":
        #         lines = lines[:-1]
        #     mermaid_code = '\n'.join(lines)

        return mermaid_code

    except ImportError:
        st.error("❌ OpenAI library not installed. Run: pip install openai")
        return None
    except Exception as e:
        st.error(f"❌ OpenAI API error: {str(e)}")
        return None


def _add_mermaid_to_session_markdown(slide_num: int, mermaid_code: str):
    """Add the generated Mermaid code to the session markdown content."""
    if 'markdown_content' not in st.session_state:
        st.warning("⚠️ No markdown content found in session")
        return

    # Find the slide in the markdown and add the Mermaid code
    markdown_content = st.session_state.markdown_content

    # Look for the slide marker
    slide_marker = f"<!-- Slide {slide_num} -->"

    if slide_marker in markdown_content:
        # Find the next slide or end of content
        lines = markdown_content.split('\n')
        insert_position = None

        for i, line in enumerate(lines):
            if slide_marker in line:
                # Find where to insert (before next slide or at end)
                insert_position = i + 1
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith('<!-- Slide '):
                        insert_position = j
                        break
                else:
                    insert_position = len(lines)
                break

        if insert_position is not None:
            # Insert the Mermaid code
            mermaid_section = [
                "",
                f"### 🎯 AI-Generated Diagram for Slide {slide_num}",
                "",
                "```mermaid",
                mermaid_code,
                "```",
                ""
            ]

            # Insert at the found position
            lines[insert_position:insert_position] = mermaid_section

            # Update session state
            st.session_state.markdown_content = '\n'.join(lines)

            st.success(f"✅ Added Mermaid diagram to Slide {slide_num} in markdown content!")
        else:
            st.error(f"❌ Could not find position to insert diagram for Slide {slide_num}")
    else:
        st.error(f"❌ Slide {slide_num} not found in markdown content")


def _clear_screenshot_results():
    """Clear all screenshot and AI analysis results."""
    keys_to_clear = ['screenshot_results', 'screenshot_slide_numbers']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Clear all AI analysis results
    keys_to_remove = []
    for key in st.session_state.keys():
        if key.startswith('ai_analysis_'):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del st.session_state[key]


def store_uploaded_file_data(uploaded_file):
    """Store PowerPoint file data for screenshot functionality."""
    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower() if '.' in uploaded_file.name else ''
        if file_ext in ['pptx', 'ppt']:
            st.session_state.uploaded_file_data = uploaded_file.getbuffer()
            st.session_state.uploaded_file_name = uploaded_file.name


def clear_screenshot_data():
    """Clear screenshot-related session data."""
    keys_to_clear = ['uploaded_file_data', 'uploaded_file_name']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def debug_screenshot_info():
    """Display debug information about screenshot functionality."""
    with st.expander("🐛 Screenshot Debug Info"):
        st.markdown("**Session State:**")
        for key in ['markdown_content', 'file_name', 'uploaded_file_data']:
            has_key = key in st.session_state
            st.write(f"• {key}: {'✅' if has_key else '❌'}")

            if has_key and key == 'uploaded_file_data':
                size_mb = len(st.session_state[key]) / (1024 * 1024)
                st.write(f"  Size: {size_mb:.2f} MB")

        # LibreOffice status
        st.markdown("**LibreOffice Status:**")
        from src.processors.diagram_screenshot_processor import test_diagram_screenshot_capability
        available, status = test_diagram_screenshot_capability()
        st.write(f"• Status: {status}")
        st.write(f"• Available: {'✅' if available else '❌'}")

