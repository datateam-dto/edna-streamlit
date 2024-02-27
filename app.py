import streamlit as st
from operator import itemgetter

from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import yaml

load_dotenv()  # Take environment variables from .env.

# streamlit_app.py

import hmac
import streamlit as st


# def check_password():
#     """Returns `True` if the user had the correct password."""

#     def password_entered():
#         """Checks whether a password entered by the user is correct."""
#         if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # Don't store the password.
#         else:
#             st.session_state["password_correct"] = False

#     # Return True if the password is validated.
#     if st.session_state.get("password_correct", False):
#         return True

#     # Show input for password.
#     st.text_input(
#         "Password", type="password", on_change=password_entered, key="password"
#     )
#     if "password_correct" in st.session_state:
#         st.error("😕 Password incorrect")
#     return False


# if not check_password():
#     st.stop()  # Do not continue if check_password is not True.
    
# Instantiate the model
model = ChatOpenAI(model_name='gpt-4-0125-preview')
model_parser = model | StrOutputParser()

# Function to dynamically generate ChatPromptTemplates from UI editable prompts
def generate_prompt_template(template):
    return ChatPromptTemplate.from_template(template)

# Placeholder function for generating outputs based on custom prompts
def process_content_with_custom_prompt(content, custom_prompt_template):
    custom_prompt = generate_prompt_template(custom_prompt_template)
    custom_prompt_pass = (
    {"input_text": RunnablePassthrough()} | custom_prompt | model_parser
    )
    return custom_prompt_pass.stream(content)

# Streamlit app UI
st.title("QA Flow Processor with Customizable Prompts")
st.image('qa-flow.png')
content = st.text_area("Content", "Enter your content here", height=400)

with st.expander("See and Modify Prompts"):
    try:
        with open("prompts.yaml", "r") as file:
            prompts = yaml.safe_load(file)
    except FileNotFoundError:
        prompts = {}

    deped_standards_prompt_text = st.text_area("DepEd Standards Prompt", value=prompts.get("deped_standards_prompt_text", ""), height=300)
    class_builder_standards_prompt_text = st.text_area("Class Builder Standards Prompt", value=prompts.get("class_builder_standards_prompt_text", ""), height=300)
    academic_learning_theories_prompt_text = st.text_area("Academic Learning Theories Prompt", value=prompts.get("academic_learning_theories_prompt_text", ""), height=300)
    grammar_and_spelling_prompt_text = st.text_area("Grammar and Spelling Prompt", value=prompts.get("grammar_and_spelling_prompt_text", ""), height=300)
    theme_consistency_prompt_text = st.text_area("Theme Consistency Prompt", value=prompts.get("theme_consistency_prompt_text", ""), height=300)
    completeness_prompt_text = st.text_area("Completeness Prompt", value=prompts.get("completeness_prompt_text", ""), height=300)
    summarize_prompt_text = st.text_area("Summarize Prompt", value=prompts.get("summarize_prompt_text", ""), height=300)
    final_prompt_text = st.text_area("Final Prompt", value=prompts.get("final_prompt_text", ""), height=300)

if st.button("Save Prompts"):
    prompts = {
        "deped_standards_prompt_text": deped_standards_prompt_text,
        "grammar_and_spelling_prompt_text": grammar_and_spelling_prompt_text,
        "class_builder_standards_prompt_text": class_builder_standards_prompt_text,
        "academic_learning_theories_prompt_text": academic_learning_theories_prompt_text,
        "theme_consistency_prompt_text": theme_consistency_prompt_text,
        "completeness_prompt_text": completeness_prompt_text,
        "summarize_prompt_text": summarize_prompt_text,
        "final_prompt_text": final_prompt_text,
    }
    with open("prompts.yaml", "w") as file:
        yaml.dump(prompts, file)

if st.button("Process Content"):

    st.text("Cognitively High-Level QA")
    # Process each area with the customized prompt
    with st.expander("See DepEd Standards Output"):
        deped_standards_output = process_content_with_custom_prompt(content, deped_standards_prompt_text)
        deped_standards_output_full = st.write_stream(deped_standards_output)

    with st.expander("See Class Builder Standards Output"):
        class_builder_standards_output = process_content_with_custom_prompt(content, class_builder_standards_prompt_text)
        class_builder_standards_output_full = st.write_stream(class_builder_standards_output)

    with st.expander("See Academic Learning Theories Output"):
        academic_learning_theories_output = process_content_with_custom_prompt(content, academic_learning_theories_prompt_text)
        academic_learning_theories_output_full = st.write_stream(academic_learning_theories_output)
    
    st.text("Cognitively Low-Level QA")

    with st.expander("See Grammar and Spelling Output"):
        grammar_and_spelling_output = process_content_with_custom_prompt(content, grammar_and_spelling_prompt_text)
        grammar_and_spelling_output_full = st.write_stream(grammar_and_spelling_output)

    with st.expander("See Theme Consistency Output"):
        theme_consistency_output = process_content_with_custom_prompt(content, theme_consistency_prompt_text)
        theme_consistency_output_full = st.write_stream(theme_consistency_output)

    with st.expander("See Completeness Output"):
        completeness_output = process_content_with_custom_prompt(content, completeness_prompt_text)
        completeness_output_full = st.write_stream(completeness_output)

    st.text("Summaries of Cognitively High-Level and Cognitively Low-Level QA")

    with st.expander("See Summary Output - Cog High"):
        summarize_output_high_cog_output = process_content_with_custom_prompt(f"DepEd Standards Output:\n{deped_standards_output_full}\nClass Builder Standards Output:\n{class_builder_standards_output_full}\nAcademic Learning Theories Output:\n{academic_learning_theories_output_full}", summarize_prompt_text)
        summarize_output_high_cog_output_full = st.write_stream(summarize_output_high_cog_output)

    with st.expander("See Summary Output - Cog Low"):
        summarize_output_low_cog_output = process_content_with_custom_prompt(f"Grammar and Spelling Output:\n{grammar_and_spelling_output_full}\nTheme Consistency Output:\n{theme_consistency_output_full}\nCompleteness Output:\n{completeness_output_full}", summarize_prompt_text)
        summarize_output_low_cog_output_full = st.write_stream(summarize_output_low_cog_output)

    st.text("Proposed Final Output Based on Summaries")

    with st.expander("See Final Output"):
        final_output = process_content_with_custom_prompt(f"Cognitively High QA Summary:\n{summarize_output_high_cog_output_full}\nCognitively Low QA Summary:\n{summarize_output_low_cog_output_full}\nContent:\n{content}", final_prompt_text)
        st.write_stream(final_output)