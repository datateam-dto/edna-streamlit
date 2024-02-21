import streamlit as st
from operator import itemgetter

from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

model = ChatOpenAI(model_name='gpt-3.5-turbo')
model_parser = model | StrOutputParser()

deped_standards_prompt = ChatPromptTemplate.from_template(
    """Align to DepEd standards: {input_text}:"""
)

class_builder_standards_prompt = ChatPromptTemplate.from_template(
    """Align to class builder standards: {input_text}:"""
)

academic_learning_theories_prompt = ChatPromptTemplate.from_template(
    """Align to academic learning theories: {input_text}:"""
)

grammar_and_spelling_prompt = ChatPromptTemplate.from_template(
    """Make sure that the grammar and spelling of this text is correct: {input_text}"""
)

theme_consistency_prompt = ChatPromptTemplate.from_template(
    """Make sure the theme across this text is consistent: {input_text}"""
)

completeness_prompt = ChatPromptTemplate.from_template(
    """Make sure that the following text has the following components {input_text}:"""
)

summarize_prompt = ChatPromptTemplate.from_template(
    """Summarize {input_text}:"""
)

# Placeholder functions for each node in your diagram
def deped_standards(content):
    deped_standards_pass = (
    {"input_text": RunnablePassthrough()} | deped_standards_prompt | model_parser
    )
    return deped_standards_pass.invoke(content)

def class_builder_standards(content):
    class_builder_standards_pass = (
    {"input_text": RunnablePassthrough()} | class_builder_standards_prompt | model_parser
    )
    return class_builder_standards_pass.invoke(content)

def academic_learning_theories(content):
    academic_learning_theories_pass = (
    {"input_text": RunnablePassthrough()} | academic_learning_theories_prompt | model_parser
    )
    return academic_learning_theories_pass.invoke(content)

def grammar_and_spelling(content):
    grammar_and_spelling_pass = (
    {"input_text": RunnablePassthrough()} | grammar_and_spelling_prompt | model_parser
    )
    return grammar_and_spelling_pass.invoke(content)

def theme_consistency(content):
    theme_consistency_pass = (
    {"input_text": RunnablePassthrough()} | theme_consistency_prompt | model_parser
    )
    return theme_consistency_pass.invoke(content)

def completeness(content):
    completeness_pass = (
    {"input_text": RunnablePassthrough()} | completeness_prompt | model_parser
    )
    return completeness_pass.invoke(content)

def high_level_qa(content):
    return "\n".join([
        deped_standards(content),
        class_builder_standards(content),
        academic_learning_theories(content),
    ])

def low_level_qa(content):
    return "\n".join([
        grammar_and_spelling(content),
        theme_consistency(content),
        completeness(content),
    ])

def condensed_summary_cog_high_qa(content):
    summarize_pass = (
    {"input_text": RunnablePassthrough()} | summarize_prompt | model_parser
    )
    return summarize_pass.invoke(content)

def condensed_summary_cog_low_qa(content):
    summarize_pass = (
    {"input_text": RunnablePassthrough()} | summarize_prompt | model_parser
    )
    return summarize_pass.invoke(content)

def final_output(high_level_summary, low_level_summary):
    return "\n".join([high_level_summary, low_level_summary, "Final Output combined content"])

# Streamlit app
st.title("QA Flow Processor")
st.image('qa-flow.png')
content = st.text_area("Content", "Enter your content here")

if st.button("Process Content"):
    high_level_qa_output = high_level_qa(content)
    st.text_area("High Level QA Output", value=high_level_qa_output, height=300)

    low_level_qa_output = low_level_qa(content)
    st.text_area("Low Level QA Output", value=low_level_qa_output, height=300)

    high_level_summary = condensed_summary_cog_high_qa(high_level_qa_output)
    st.text_area("High Level Summary", value=high_level_summary, height=300)

    low_level_summary = condensed_summary_cog_low_qa(low_level_qa_output)
    st.text_area("Low Level Summary", value=low_level_summary, height=300)

    final_output_content = final_output(high_level_summary, low_level_summary)
    st.text_area("Final Output Content", value=final_output_content, height=300)


        
        
        
