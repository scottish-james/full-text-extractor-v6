"""
About tab component for the Document to Markdown Converter.
"""

import streamlit as st
from src.content.about_content import get_about_content, get_technical_benefits, get_problem_solution_pairs


def render_about_tab():
    """Render the complete about tab content."""

    # Main about content
    st.markdown(get_about_content())

    st.markdown("---")

    # Technical benefits section
    st.subheader("🚀 Key Technical Benefits")

    benefits = get_technical_benefits()
    for benefit, description in benefits.items():
        with st.expander(f"**{benefit}**"):
            st.write(description)

    st.markdown("---")

    # Problem/Solution pairs
    st.subheader("💡 Common Challenges We Solve")

    problems_solutions = get_problem_solution_pairs()

    for i, pair in enumerate(problems_solutions, 1):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"**❌ Challenge {i}:**")
            st.write(pair["problem"])

        with col2:
            st.markdown(f"**✅ Our Solution:**")
            st.write(pair["solution"])

        if i < len(problems_solutions):
            st.markdown("")  # Add spacing between pairs

    st.markdown("---")

    # Call to action
    st.info(
        "💼 **Ready to improve your AI pipeline?** Upload a document in the File Upload tab to get started, or process multiple files using Folder Processing.")


def render_compact_about():
    """Render a compact version for sidebar or small spaces."""
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        Built for production-grade AI solutions. This tool creates high-quality markdown from your documents, 
        optimised for RAG systems and LLM training.

        **Key Benefits:**
        - Improved embedding accuracy
        - Section-based chunking
        - Reading order intelligence
        - Pipeline-ready Python integration
        """)

        if st.button("Learn More", key="learn_more_compact"):
            st.info("Switch to the 'About' tab for full details!")