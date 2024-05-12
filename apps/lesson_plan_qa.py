

# External libraries
import streamlit as st
from PyPDF2 import PdfReader
import openai
import sys 
import os
import pysqlite3
import re
#from io import StringIO

# Set the path as environment variable
#sys.path.append("/mount/src/edna-streamlit/apps")


from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from streamlit_chat import message
from langchain import PromptTemplate
import st_btn_select
import tempfile
from pdfminer.high_level import extract_text
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings




import base64 # byte object into a pdf file 
#import camelot as cam # extracting tables from PDFs 


sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
openai.organization = "org-ydtCQcRROzj3YuGKoh4NtXEV"
openai_api_key = st.secrets["OPENAI_API_KEY"]



def qa_file(data):
    
    if 'chain' not in st.session_state:
        #loader = TextLoader(file_path=filepath)
        #data = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100,separator="\n")
        
        print("splitting")
        texts = text_splitter.split_documents(data)
            
            
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever(search_type = "similarity", search_kwargs = {"k":5})
        print("splitting")
        chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.1,model = 'gpt-4-turbo-2024-04-09', openai_api_key=openai_api_key),
                                                                                retriever=retriever)
    
        st.session_state['chain'] = chain 
    def conversational_chat(query):
        print("sending q")  
        chain =  st.session_state['chain']
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
                
        return result["answer"]
            
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about the lesson plan 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
                
        #container for the chat history
    response_container = st.container()
        #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Ask your question here (:", key='input')
            submit_button = st.form_submit_button(label='Send', on_click=None)
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                    
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i),avatar_style="initials", seed = "EDNA")


def split_text(text):
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    docs = text_splitter.create_documents([text])
    print(docs[0].page_content)
    st.text(docs[0].page_content)

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

def convert_to_markdown(text):
    markdown = text
    markdown = re.sub(r'^(#+)(.*)', r'\1 \2', markdown, flags=re.MULTILINE)
    markdown = re.sub(r'([I|II|III|IV|V|VI|VII]+\.)\s+(.*)', r'# \1 \2', markdown, flags=re.MULTILINE)
    # Convert ordered lists
    markdown = re.sub(r'^\s*([A-H]+\.)\s+(.*)\n(.*)', r'## \1 \2 \3', markdown, flags=re.MULTILINE)
    # Convert unordered lists
    markdown = re.sub(r'^\*\s+(.*)', r'- \1', markdown, flags=re.MULTILINE)
    
    return markdown

def main():
    # Start of streamlit application
    st.title("Lesson Plan QA Bot")


    # Intitialization
    st.header("File upload")
    uploaded_file = st.file_uploader("Choose a file (pdf)", type=["pdf"], help="file to be parsed")
    if uploaded_file is not None :
        #html_output  = extract_text_to_fp(uploaded_file, output_string, laparams=LAParams(),output_type='html', codec=None)
        #content = extract_text(uploaded_file, page_numbers=None, maxpages=0, caching=True, codec='utf-8', laparams=None)
        #print(text)
        content = extract_text_(uploaded_file)
        #md_text = convert_to_markdown(content)
        st.markdown(content)
        st.text(content)
        #st.markdown(md_text)
        #qa_file(content)
        #split_text(content)

if __name__ == "__main__":
 
    main()