import os
import streamlit as st
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import requests
from bs4 import BeautifulSoup
import random
from chardet import detect

def check_encoding(file: str = None)-> str:
    """
    Args:
        file: str, name of the file to check
    Returns:
        encoding: str, detected most probable encoding
    """
    with open(os.path.join("data", file), 'rb') as f:
        encoding = detect(f.read(10000))['encoding']
        return encoding
def get_random_quote():
    """
    Pobiera losowy cytat ze strony Wikiquote "Władca Pierścieni".
    Returns:
        str: Losowy cytat jako łańcuch znaków.
        None: Jeśli nie uda się znaleźć cytatów.
    """
    # URL strony z cytatami
    url = 'https://pl.wikiquote.org/wiki/Władca_Pierścieni'
    try:
        # Pobranie zawartości strony
        response = requests.get(url)
        response.raise_for_status()  # Sprawdzenie, czy żądanie się powiodło

        # Parsowanie HTML za pomocą BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Znalezienie wszystkich elementów <li> w sekcji z cytatami
        quote_elements = soup.select('div#mw-content-text ul li')

        # Wyodrębnienie tekstu z każdego elementu <li>
        quotes = [quote.get_text(strip=True) for quote in quote_elements]

        # Usunięcie pustych cytatów
        quotes = [quote for quote in quotes if quote]

        # Sprawdzenie, czy znaleziono cytaty
        if not quotes:
            return None

        # Losowy wybór cytatu
        return random.choice(quotes)

    except requests.RequestException as e:
        print(f"Błąd w czasie pobierania strony: {e}")
        return None

logger = get_logger('Langchain-Chatbot')

#decorator
def enable_chat_history(func):
    if os.environ.get("OPENAI_API_KEY"):

        # to clear chat history after swtching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # to show chat history on ui
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Co chciałbyś wiedzieć?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def configure_llm():
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0, streaming=True)
    return llm

def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))

@st.cache_resource
def configure_embedding_model():
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embedding_model

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v

import os
import pickle
from langchain_community.vectorstores import DocArrayInMemorySearch

import os
import joblib
from langchain_community.vectorstores import DocArrayInMemorySearch

import numpy as np

def save_vector_data(vector_db, file_path):
    """
    Save vector data and metadata to a file.
    Args:
        vector_db: DocArrayInMemorySearch object to save.
        file_path: Path where the database should be saved.
    """
    vectors = [doc.embedding for doc in vector_db]
    metadata = [doc.metadata for doc in vector_db]
    np.savez(file_path, vectors=vectors, metadata=metadata)

def load_vector_data(file_path, embedding_model):
    """
    Load vector data and recreate DocArrayInMemorySearch.
    Args:
        file_path: Path of the saved data.
        embedding_model: Embedding model for recreating the database.
    Returns:
        DocArrayInMemorySearch object.
    """
    data = np.load(file_path, allow_pickle=True)
    vectors = data['vectors']
    metadata = data['metadata']

    # Create documents from vectors and metadata
    documents = [
        Document(page_content="", metadata=meta, embedding=vec)
        for vec, meta in zip(vectors, metadata)
    ]

    return DocArrayInMemorySearch.from_documents(documents, embedding_model)




def create_or_load_vector_db(docs, embedding_model, file_path="vector_db.npz"):
    """
    Create or load a vector database using NumPy format.

    Args:
        docs: List of documents to vectorize if database doesn't exist.
        embedding_model: Embedding model for creating vector embeddings.
        file_path: Path to save/load the vector database.

    Returns:
        DocArrayInMemorySearch object.
    """
    if os.path.exists(file_path):
        # Load the database if the file exists
        print(f"Loading vector database from {file_path}...")
        data = np.load(file_path, allow_pickle=True)
        vectors = data['vectors']
        metadata = data['metadata']

        # Recreate documents
        documents = [
            Document(page_content="", metadata=meta, embedding=vec)
            for vec, meta in zip(vectors, metadata)
        ]

        # Create the vector database
        return DocArrayInMemorySearch.from_documents(documents, embedding_model)
    else:
        # If the file does not exist, create a new database
        print(f"Creating a new vector database and saving to {file_path}...")
        # Vectorize documents
        vectors = []
        metadata = []
        for doc in docs:
            embedding = embedding_model.embed_query(doc.page_content)
            vectors.append(embedding)
            metadata.append(doc.metadata)

        # Save data to a .npz file
        np.savez(file_path, vectors=vectors, metadata=metadata)

        # Create the vector database
        documents = [
            Document(page_content="", metadata=meta, embedding=vec)
            for vec, meta in zip(vectors, metadata)
        ]
        return DocArrayInMemorySearch.from_documents(documents, embedding_model)


from langchain_core.callbacks import BaseCallbackHandler
class StreamHandler(BaseCallbackHandler):

    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

#import utils
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

st.set_page_config(page_title="Chat", page_icon="📄")
st.header('                 Pal-AI-ntir - Twój osobisty przewodnik po Śródziemiu. ')
st.write('\n\n              Powiedz przyjacielu i pytaj')

class CustomDocChatbot:

    def __init__(self):
        sync_st_session()
        self.llm = configure_llm()
        self.embedding_model = configure_embedding_model()

    @st.spinner(f'Spokojnie...Czat nigdy się nie spóźnia ani nie śpieszy lecz odpowiada właśnie wtedy kiedy uzna to za właściwe.\n W czasie oczekiwania przemyśl te słowa: \n"{get_random_quote()}"')
    def import_source_documents(self):
        # Load documents
        docs = []
        files = []
        for file in os.listdir("data"):
            if file.endswith(".txt"):
                encoding = check_encoding(file)
                try:
                    with open(os.path.join("data", file),
                              encoding=encoding) as f:
                        docs.append(os.path.join("data", f.read()))
                        files.append(file)
                except Exception as e:
                    print(file)
                    continue

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = []
        for i, doc in enumerate(docs):
            for chunk in text_splitter.split_text(doc):
                splits.append(Document(page_content=chunk, metadata={"source": files[i]}))

        # Create or load vector database
        vectordb_file_path = "vector_db.npz"
        # Path to save the database
        vectordb = create_or_load_vector_db(splits, self.embedding_model, vectordb_file_path)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 2, 'fetch_k': 4}
        )

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            """
            You are a chatbot tasked with responding to questions based on attached websites content below.
            {context}
            
            Considering text above answer following question. Deepend only on source documents.
            {question}
            """
        )

        prompt = ChatPromptTemplate.from_messages([system_message_prompt])

        # Setup LLM and QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

        return qa_chain


    @enable_chat_history
    def main(self):
        user_query = st.chat_input(placeholder="Ask for information from documents")

        if user_query:
            qa_chain = self.import_source_documents()

            display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())

                result = qa_chain.invoke(
                    {"question":user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                print_qa(CustomDocChatbot, user_query, response)

                # to show references
                for  doc in result['source_documents']:
                    filename = os.path.basename(doc.metadata['source'])
                    ref_title = f":blue[Source document: {filename}]"
                    with st.popover(ref_title):
                        st.caption(doc.page_content)

if __name__ == "__main__":
    obj = CustomDocChatbot()
    obj.main()