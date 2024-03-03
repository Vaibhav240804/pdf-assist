import pickle
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os

load_dotenv()


PDF_PATH = "pdfs"  # Replace with the actual path to your fixed PDFs
TEXT_FILE_PATH = "txt/extracted_text.txt"  # Replace with the desired path for the text file

# Function to extract text from the specified PDFs and save it to a text file
def extract_and_save_text():
    text = ""
    for filename in os.listdir(PDF_PATH):
        if filename.endswith(".pdf"):
            pdf_file = os.path.join(PDF_PATH, filename)
            with open(pdf_file, "rb") as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text()
    with open(TEXT_FILE_PATH, "w") as f:
        print(f"Saving text to {TEXT_FILE_PATH}")
        f.write(text)

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512},huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"))
    llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.5, "max_length":512},huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"))

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    print("Handling user input")
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    # Load text from the text file if it exists, otherwise extract and save it
    if os.path.exists(TEXT_FILE_PATH):
        print("Text file exists")
        with open(TEXT_FILE_PATH, "r") as f:
            raw_text = f.read()
    else:
        print("Text file does not exist")
        extract_and_save_text()
        with open(TEXT_FILE_PATH, "r") as f:
            raw_text = f.read()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # with st.sidebar:
    #     st.subheader("Your documents")
    #     pdf_docs = st.file_uploader(
    #         "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    #     if st.button("Process"):
    #         with st.spinner("Processing"):
    #             # get pdf text
    #             raw_text = get_pdf_text(pdf_docs)

                # get the text chunks

    # create vector store
    try:
        with open("vectorstore.pkl", "rb") as f:
            print("Loading vectorstore from file")
            vectorstore = pickle.load(f)
            print("Loaded vectorstore from file")
    except FileNotFoundError:
        print("Creating vectorstore")
        try:
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            with open("vectorstore.pkl", "wb") as f:
                pickle.dump(vectorstore, f)
            print("Saved vectorstore to file")
        except Exception as e:
            print("Error creating vectorstore")
            print(e)

    # create conversation chain
    st.session_state.conversation = get_conversation_chain(
        vectorstore)


if __name__ == '__main__':
    main()
