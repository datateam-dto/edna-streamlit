import streamlit as st
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import yaml

load_dotenv()  # Load environment variables from .env.

# Initialize the model
model = ChatOpenAI(model_name='gpt-4-0125-preview')
model_parser = model | StrOutputParser()

# Dynamically generate ChatPromptTemplates from user input
def generate_prompt_template(template):
    return ChatPromptTemplate.from_template(template)

# Generate outputs based on custom prompts
def process_content_with_custom_prompt(content, custom_prompt_template):
    if custom_prompt_template.strip() == "":
        return "No prompt provided."
    custom_prompt = generate_prompt_template(custom_prompt_template)
    custom_prompt_pass = (
        {"input_text": RunnablePassthrough()} | custom_prompt | model_parser
    )
    return custom_prompt_pass.stream(content)

# Function to compile selected outputs for use in subsequent prompts
def compile_outputs(outputs):
    compiled_content = "\n".join(
        f"Output {i+1}: {outputs[i]}" for i in outputs
    )
    return compiled_content

# Streamlit UI
st.title("QA Flow Processor with Customizable Prompts")

content = st.text_area("Content", "Enter your content here", height=400)

try:
    with open("prompts.yaml", "r") as file:
        prompts = yaml.safe_load(file)
except FileNotFoundError:
    prompts = {}

outputs = {}
prompt_texts = []

with st.expander("See and Modify Prompts"):
    for i in range(6):
        # Dynamically create checkboxes and store their state in session_state
        check = st.checkbox(f"Use Preprocess Prompt {i+1}", value=True, key=f'check_{i+1}')
        prompt_text = st.text_area(f"Preprocess Prompt {i+1}", value=prompts.get(f"preprocess_prompt_{i+1}_text", ""), height=300, key=f'prompt_{i+1}')
        prompt_texts.append(prompt_text)

    chain_output_1_check = st.checkbox("Use Chain Output 1", value=True, key='chain_output_1_check')
    chain_output_1_text = st.text_area("Chain Output 1 Prompt", value=prompts.get("chain_output_1_text", ""), height=300, key='chain_output_1_text')

    chain_output_2_check = st.checkbox("Use Chain Output 2", value=True, key='chain_output_2_check')
    chain_output_2_text = st.text_area("Chain Output 2 Prompt", value=prompts.get("chain_output_2_text", ""), height=300, key='chain_output_2_text')

# Process content and prompts
if st.button("Process Content"):
    for i, prompt_text in enumerate(prompt_texts):
        # Check the current state directly using session_state
        if st.session_state[f'check_{i+1}']:
            output_stream = process_content_with_custom_prompt(content, prompt_text)
            
            with st.expander(f"Preprocess Prompt {i+1} Output"):
                output = st.write_stream(output_stream)
                outputs[i] = output

    compiled_outputs = compile_outputs(outputs)

    if chain_output_1_check:
        output_1 = process_content_with_custom_prompt(compiled_outputs, chain_output_1_text)
        with st.expander("Chain Output 1"):
            st.write(output_1)

    if chain_output_2_check:
        output_2 = process_content_with_custom_prompt(compiled_outputs, chain_output_2_text)
        with st.expander("Chain Output 2"):
            st.write(output_2)
