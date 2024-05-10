# External libraries
import streamlit as st
from streamlit_chat import message
from PyPDF2 import PdfReader
import openai
import sys 
import os

# Set the path as environment variable
os.chdir("/apps/")
sys.path.append("/apps/")

# Internal file imports
from extract import extract_text

# from qa import search_qa
from chat_pdf import chat_with_pdf

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
import tempfile

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
openai.organization = "org-ydtCQcRROzj3YuGKoh4NtXEV"
openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = OpenAI(temperature=0.1)
openai.Model.list()


def qa_file(filepath):
    
    if 'chain' not in st.session_state:
        loader = TextLoader(file_path=filepath)
        data = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100,separator="?")
        
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

def main():
    # Start of streamlit application
    st.title("Lesson Plan QA Bot")

    # Intitialization
    st.header("File upload")
    uploaded_txt_file = st.file_uploader("Choose a file (docx, txt)", type="pdf", help="file to be parsed")
    if uploaded_txt_file is not None :
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_txt_file.getvalue())
                    tmp_file_path = tmp_file.name
                    qa_file(tmp_file_path)

if __name__ == "__main__":
 
    main()