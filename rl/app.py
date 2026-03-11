import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from rl.streamlit.styles import apply_page_config, inject_custom_styles

VIEWS = {
    "Home": {
        "module": "rl.streamlit.views.home",
        "icon": ":material/home:",
    },
    "Datasets": {
        "module": "rl.streamlit.views.dataset_viewer",
        "icon": ":material/dataset:",
        "title": "Datasets",
        "description": "Browse negotiation conversations and compare generated annotations.",
    },
    "Training": {
        "module": "rl.streamlit.views.training_viewer",
        "icon": ":material/show_chart:",
        "title": "Training",
        "description": "Visualize SFT training loss and metrics over time.",
    },
}


def _set_active_view(view_name: str):
    st.session_state["active_view"] = view_name


def _render_sidebar_view_switcher() -> str:
    if (
        "active_view" not in st.session_state
        or st.session_state["active_view"] not in VIEWS
    ):
        st.session_state["active_view"] = "Home"

    current_view = st.session_state["active_view"]
    for view_name, meta in VIEWS.items():
        st.sidebar.button(
            view_name,
            icon=meta["icon"],
            key=f"view-nav-{view_name}",
            use_container_width=True,
            type="primary" if view_name == current_view else "secondary",
            on_click=_set_active_view,
            args=(view_name,),
        )

    return st.session_state["active_view"]


def main():
    apply_page_config()
    inject_custom_styles()

    view_name = _render_sidebar_view_switcher()
    meta = VIEWS[view_name]

    if "title" in meta:
        st.markdown(f"### {meta['title']}")
        st.markdown(
            f"<p class='view-description'>{meta['description']}</p>",
            unsafe_allow_html=True,
        )

    mod = importlib.import_module(meta["module"])
    mod.render()


if __name__ == "__main__":
    main()
