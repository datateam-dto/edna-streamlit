

# External libraries
import streamlit as st
from PyPDF2 import PdfReader
import openai
import sys 
import os
import pysqlite3
import re
import yaml

# Set the path as environment variable
#sys.path.append("/mount/src/edna-streamlit/apps")



from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
import st_btn_select
import tempfile

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory






import base64 # byte object into a pdf file 
#import camelot as cam # extracting tables from PDFs 


#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
openai.organization = "org-ydtCQcRROzj3YuGKoh4NtXEV"
openai_api_key = st.secrets["OPENAI_API_KEY"]

try:
    with open("prompts_.yaml", "r") as file:
        prompts = yaml.safe_load(file)
except FileNotFoundError:
    prompts = {}


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")
    def on_retriever_end(self, documents, **kwargs):
        self.status.update(state="complete")



def qa_file(splits):
    general_system_template = """You are a Quality Assurance assistant, checking the quality of the lesson plan for a course.  
    The user will provide you with a set of documents which is a lesson plan for a course.
    You are tasked to answer questions on the lesson plan.
    You may need to look at the whole Lesson plan to provide your answers.
    
    {context}
    """
    general_user_template ="{question}"
    messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]
    qa_prompt = ChatPromptTemplate.from_messages( messages )

    embeddings = OpenAIEmbeddings()
    #db = Chroma.from_documents(splits, embeddings)
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20})

    
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    

    if 'chain' not in st.session_state:
            # Setup LLM and QA chain
        llm = ChatOpenAI(
            model = "gpt-4-turbo-2024-04-09", openai_api_key=openai_api_key, temperature=.2, streaming=True
            )
        chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": qa_prompt})
        st.session_state['chain'] = chain 

    chain = st.session_state['chain']
   
    if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
            msgs.clear()
            msgs.add_ai_message("How can I help you?")

    avatars = {"human": "user", "ai": "👩🏻‍🏫"}

    
    #if user_query := st.chat_input(placeholder="Ask me anything!"):
      #  st.chat_message("user").write(user_query)
      #  with st.chat_message("assistant"):
      #      retrieval_handler = PrintRetrievalHandler(st.container())
      #      stream_handler = StreamHandler(st.empty())
       #     response = chain.run({"question": user_query}, callbacks=[stream_handler, retrieval_handler])
        
        #container for the chat history
    response_container = st.container()
        #container for the user's text input
    container = st.container()
    with response_container:
        for msg in msgs.messages:
            st.chat_message(avatars[msg.type]).write(msg.content)

    with container:
        init_prompt = st.selectbox(
        'You might want to try these prompts, click to expand',
        prompts.get("lesson_plan").keys())

        instr = 'Hi there! "Ask your question here.'
        with st.form(key='my_form', clear_on_submit=True):
            user_query = st.text_input(instr,value=prompts.get("lesson_plan")[init_prompt],placeholder=instr, label_visibility='collapsed')
            submit_button = st.form_submit_button(label='Send', on_click=None)
            if submit_button and user_query:
                with response_container:
                    st.chat_message("user").write(user_query)
                    with st.chat_message("assistant", avatar = "👩🏻‍🏫"):
                        retrieval_handler = PrintRetrievalHandler(st.container())
                        stream_handler = StreamHandler(st.empty())
                        response = chain.run({"question": user_query}, callbacks=[stream_handler, retrieval_handler])
     
    

def split_text_semantic(text):
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    docs = text_splitter.create_documents([text])
    print(docs[0].page_content)
    st.text(docs[0].page_content)

def split_splits_md(md_splits):
    chunk_size = 1000
    chunk_overlap = 150
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(md_splits)
    return splits


def split_splits_semantic(md_splits):
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    splits = text_splitter.create_documents(md_splits)
    return splits

def split_text_markdown(markdown_document):
    headers_to_split_on = [
    ("#", "Section"),
    ("##", "Part"),]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    md_header_splits = markdown_splitter.split_text(markdown_document)
    #st.write(md_header_splits)
    splits = split_splits_md(md_header_splits)
    return splits

def extract_text_(_file):
    """
        :param file: the PDF file to extract
    """
    content = ""
    reader = PdfReader(_file)
    number_of_pages = len(reader.pages)

    # Scrape text from multiple pages
    for i in range(number_of_pages):
        page = reader.pages[i]
        text = page.extract_text()
        content = content + text
    return content

def convert_to_markdown(text):
    markdown = text
    markdown = re.sub(r'^(#+)(.*)', r'\1 \2', markdown, flags=re.MULTILINE)
    markdown = re.sub(r'([IV|VI|VII|III|II|I|V]+\.)\s+(.*)', r'# \1 \2', markdown, flags=re.MULTILINE)
    # Convert ordered lists
    markdown = re.sub(r'^\s*([a-z]\))\s+(.*)\n(.*)', r'## \1 \2 \3', markdown, flags=re.MULTILINE)
    # Convert unordered lists
    markdown = re.sub(r'^\*\s+(.*)', r'- \1', markdown, flags=re.MULTILINE)
    
    return markdown

def main():
    # Start of streamlit application
    st.header("Lesson Plan QA Bot", divider = "green")


    # Intitialization
    with st.sidebar:
        st.markdown("#### File upload")
        uploaded_file = st.file_uploader("Choose a file (pdf)", type=["pdf"], help="file to be parsed")
    if uploaded_file is not None :
        content = extract_text_(uploaded_file)
        md_text = convert_to_markdown(content)  
        splits = split_text_markdown(md_text)
        qa_file(splits)
            #split_text(content)

if __name__ == "__main__":
 
    main()