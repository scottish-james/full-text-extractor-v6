"""
Updated Sidebar with OpenAI API Key Storage
Updates to src/ui/sidebar.py - stores OpenAI key in session state
"""

import streamlit as st


def setup_enhanced_sidebar():
    """Enhanced sidebar setup with Claude Sonnet 4 and OpenAI integration."""
    with st.sidebar:
        # Enhancement options
        st.subheader("AI Enhancement")
        enhance_markdown = st.toggle("Enhance with Claude Sonnet 4", value=True,
                                     help="Use Claude to improve markdown formatting and structure")

        api_key_claude = None
        if enhance_markdown:
            api_key_claude = st.text_input(
                "Anthropic API Key",
                type="password",
                help="Enter your Anthropic API key for Claude enhancement"
            )

            # Clear instructions for getting API key
            if not api_key_claude:
                st.info("💡 **Need an API key?**")
                st.markdown("""
                1. Visit [console.anthropic.com](https://console.anthropic.com/)
                2. Create an account or sign in
                3. Generate a new API key
                4. Paste it above to unlock AI enhancement
                """)
            else:
                st.success("✅ Claude Sonnet 4 ready!")

        # OpenAI Integration for Diagram Analysis
        st.subheader("Diagram Analysis")
        enhance_diagram = st.toggle("AI Diagrams using OpenAI", value=True,
                                    help="Use OpenAI to analyze screenshots and create Mermaid diagrams")

        api_key_openai = None
        if enhance_diagram:
            api_key_openai = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key for diagram analysis",
                key="openai_api_key_input"
            )

            # Store in session state for use by other components
            if api_key_openai:
                st.session_state.openai_api_key = api_key_openai
            elif 'openai_api_key' in st.session_state:
                # Clear if no key provided
                del st.session_state.openai_api_key

            # Clear instructions for getting API key
            if not api_key_openai:
                st.info("💡 **Need an API key?**")
                st.markdown("""
                1. Visit [platform.openai.com](https://platform.openai.com/)
                2. Sign in and go to **API Keys**
                3. Click **Create new secret key**
                4. Copy and save it securely
                """)
            else:
                st.success("✅ OpenAI ready for diagram analysis!")

                # Show current usage info
                with st.expander("ℹ️ OpenAI Usage Info"):
                    st.markdown("""
                    **Model Used:** GPT-4o (Vision)
                    **Purpose:** Analyze diagram screenshots and generate Mermaid code
                    **Cost:** ~$0.01-0.03 per image analysis
                    **Features:**
                    - High-quality image analysis
                    - Automatic diagram type detection
                    - Clean Mermaid code generation
                    """)

        # Developer info
        st.markdown("---")
        st.markdown("""
        **Developed by:** James Taylor  
        **Powered by:** Claude Sonnet 4 & OpenAI GPT-4o  
        """)

    return enhance_markdown, api_key_claude, api_key_openai