from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="EDNA AI Home",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from st_pages import Page, add_page_title, show_pages

add_page_title()
show_pages(
        [
            Page("main_app.py","EDNA AI Home","🛠️"),
            # Can use :<icon-name>: or the actual icon
            Page("apps/lesson_plan_qa.py", "Lesson Plan QA","✏️"),
            # The pages appear in the order you pass them
            Page("apps/lesson_plan_qa.py", "Slides QA","💡"),

        ]
    )


st.page_link("main_app.py", label="Home", icon="🛠️")
st.page_link("apps/lesson_plan_qa.py", label="Lesson Plan QA", icon="✏️")
st.page_link("apps/lesson_plan_qa.py", label = "Slides QA", icon="💡")