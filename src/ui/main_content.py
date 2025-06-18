"""
Main content UI components for the Document to Markdown Converter.
"""

import streamlit as st
from src.content.features import get_main_features, get_feature_tagline, get_tool_description


def render_main_features():
    """Render the main features section using content from features.py"""
    st.markdown(get_tool_description())
    st.markdown("")  # Add spacing

    features = get_main_features()
    feature_text = ""
    for feature_key, feature_data in features.items():
        feature_text += f"**{feature_data['icon']} {feature_data['title']}:** {feature_data['description']}  \n"

    st.markdown(feature_text)


def render_app_header():
    """Render the main application header"""
    st.title("Document to Markdown Converter")
    st.subheader(get_feature_tagline())


def render_feature_highlights():
    """Render feature highlights in a more visual way"""
    features = get_main_features()

    col1, col2, col3 = st.columns(3)

    feature_list = list(features.values())

    with col1:
        if len(feature_list) > 0:
            feature = feature_list[0]
            st.markdown(f"### {feature['icon']} {feature['title']}")
            st.markdown(feature['description'])

    with col2:
        if len(feature_list) > 1:
            feature = feature_list[1]
            st.markdown(f"### {feature['icon']} {feature['title']}")
            st.markdown(feature['description'])

    with col3:
        if len(feature_list) > 2:
            feature = feature_list[2]
            st.markdown(f"### {feature['icon']} {feature['title']}")
            st.markdown(feature['description'])


def render_welcome_section():
    """Render a complete welcome section with features"""
    render_app_header()
    render_main_features()
