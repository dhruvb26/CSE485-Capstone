import streamlit as st
from PIL import Image

PAGE_TITLE = "ASU AI - Negotiation Agent Streamlit"
PAGE_LAYOUT = "wide"
SYMBOL_PATH = "rl/assets/asu_symbol.png"


def _square_icon(path: str, size: int = 64) -> Image.Image:
    logo = Image.open(path).convert("RGBA")
    logo.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    x = (size - logo.width) // 2
    y = (size - logo.height) // 2
    canvas.paste(logo, (x, y), logo)
    return canvas


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"


def apply_page_config():
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout=PAGE_LAYOUT,
        page_icon=_square_icon(SYMBOL_PATH),
    )


def inject_custom_styles():
    primary = st.get_option("theme.primaryColor")
    bg = st.get_option("theme.backgroundColor")
    surface = st.get_option("theme.secondaryBackgroundColor")
    text = st.get_option("theme.textColor")
    border = st.get_option("theme.borderColor")

    st.markdown(
        f"""
        <style>
            :root {{
                --theme-primary: {primary};
                --theme-bg: {bg};
                --theme-text: {text};
                --theme-surface: {surface};
                --theme-border: {border};
                --theme-radius: 0.425rem;
                --theme-primary-rgb: {_hex_to_rgb(primary)};
                --theme-surface-rgb: {_hex_to_rgb(surface)};
                --theme-text-rgb: {_hex_to_rgb(text)};
                --theme-border-rgb: {_hex_to_rgb(border)};
            }}

            .block-container {{
                padding-top: 4rem;
                padding-bottom: 4rem;
            }}

            .main h1, .main h2, .main h3, .main h4 {{
                letter-spacing: -0.02em;
            }}

            .main h3 {{
                font-size: 1.35rem;
                margin-bottom: 0.6rem;
            }}

            .main h4 {{
                font-size: 1rem;
                margin-bottom: 0.7rem;
            }}

            .main p, .main li, .main label, .stMarkdown, .stCaption {{
                font-size: 0.9rem;
            }}

            .main code,
            .main code span,
            .stMarkdown code {{
                color: var(--theme-primary);
            }}

            .ann-card {{
                border-radius: var(--theme-radius);
                padding: 0.65rem 0.8rem;
                margin-bottom: 0.55rem;
                border: 1px solid var(--theme-border);
            }}

            .ann-system {{
                background: var(--theme-surface);
                border-style: dashed;
                padding: 0.5rem 2rem 0.25rem 2rem;
            }}

            .ann-system summary {{
                cursor: pointer;
                list-style: disclosure-closed;
            }}

            .ann-system details[open] > summary {{
                list-style: disclosure-open;
            }}

            .ann-system .ann-body {{
                font-size: 0.78rem;
                line-height: 1.5;
                color: rgba(var(--theme-text-rgb), 0.65);
                white-space: pre-wrap;
            }}

            .ann-user {{
                background: var(--theme-bg);
                border-color: var(--theme-border);
            }}

            .ann-assistant {{
                background: rgba(var(--theme-primary-rgb), 0.08);
                border-color: rgba(var(--theme-primary-rgb), 0.3);
            }}

            .ann-role {{
                font-size: 0.66rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: rgba(var(--theme-text-rgb), 0.52);
                margin-bottom: 0.35rem;
            }}

            .ann-body {{
                font-size: 0.84rem;
                line-height: 1.5;
                color: var(--theme-text);
                white-space: pre-wrap;
            }}

            .ann-section {{
                margin-bottom: 0.3rem;
                font-size: 0.84rem;
                line-height: 1.5;
            }}

            .ann-section:last-child {{
                margin-bottom: 0;
            }}

            .ann-section-label {{
                font-weight: 700;
                font-size: 0.72rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                color: rgba(var(--theme-text-rgb), 0.5);
                margin-right: 0.15rem;
            }}

            .ann-section-text {{
                color: var(--theme-text);
                white-space: pre-wrap;
            }}

            .muted-info {{
                color: rgba(var(--theme-text-rgb), 0.72);
                font-size: 0.92rem;
                line-height: 1.5;
                padding: 0.5rem 0;
            }}

            .home-hero-wrap {{
                max-width: 56rem;
                margin: 0 auto;
            }}

            .home-hero-title {{
                margin-top: 1.25rem;
                text-align: center;
                color: var(--theme-text);
                font-size: clamp(1.8rem, 2.35vw, 2.6rem);
                line-height: 1.16;
                text-wrap: balance;
            }}

            .home-hero-subtitle {{
                margin-top: 0.6rem;
                text-align: center;
                color: rgba(var(--theme-text-rgb), 0.52);
                font-size: 0.95rem;
                letter-spacing: 0.03em;
            }}

            .view-description {{
                margin-top: 0.3rem;
                color: rgba(var(--theme-text-rgb), 0.52);
                font-size: 0.9rem;
                line-height: 1.45;
            }}

            [data-testid="stSidebar"] .stButton > button {{
                justify-content: flex-start;
                border-radius: var(--theme-radius);
                min-height: 2.1rem;
                padding: 0.38rem 0.6rem;
                border: none;
                background: transparent;
                color: rgba(var(--theme-text-rgb), 0.72);
                box-shadow: none;
                font-size: 0.88rem;
            }}

            [data-testid="stSidebar"] .stButton > button > div {{
                justify-content: flex-start;
                text-align: left;
            }}

            [data-testid="stSidebar"] .stButton > button:hover {{
                background: var(--theme-surface);
                color: var(--theme-text);
            }}

            [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
                background: var(--theme-surface);
                color: var(--theme-text);
            }}

            [data-testid="stJson"] {{
                background: transparent !important;
                background-color: transparent !important;
                border: 1px solid var(--theme-border);
                border-radius: var(--theme-radius);
                padding: 0.5rem;
            }}

        </style>
        """,
        unsafe_allow_html=True,
    )
