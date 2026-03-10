import streamlit as st


def render():
    """Render the landing page."""
    left, center, right = st.columns([0.55, 1.9, 0.55])

    with center:
        st.markdown(
            """
            <div class="home-hero-wrap">
                <div class="home-hero-title">
                    AI for Business: Creating Smart Business Negotiations Bots
                </div>
                <div class="home-hero-subtitle">
                    CSE 486 Capstone Project II
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
