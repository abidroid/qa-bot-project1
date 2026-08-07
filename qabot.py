import os
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
 
import gradio as gr

load_dotenv()  # reads .env into environment vars


# PDFs aren't plain text
# they encode layout, fonts, and positioning. 
# PyPDFLoader handles the messy extraction 
# so you get clean, usable text.

def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document


# LLMs and embedding models work best on small pieces of text, 
# not entire documents. Splitting also lets the retriever pull only 
# the relevant chunk instead of the whole PDF.
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    return chunks


# A small, fast, well-tested sentence-embedding model. 
# First run downloads it (~90MB) automatically; 
# after that it's cached locally.


# A numeric vector (a list of numbers) that represents 
# the meaning of a chunk of text. Similar meanings end up 
# as similar vectors — that's what makes searching by
# meaning possible instead of just matching exact words.


# Important: Claude has no embeddings API — Anthropic only offers chat models. 
# That's why this piece uses a separate, 
# free HuggingFace model that runs entirely on your machine — no API key, no cost.

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name=
          "sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


# A library where every book is filed by meaning 
# instead of alphabetically — so “books about similar topics” 
# sit near each other, and you can instantly find neighbors of any given idea.
def vector_database(chunks):
    embedding_model = get_embedding_model()
    vectordb = Chroma.from_documents(
        chunks, embedding_model)
    return vectordb


# Retrieval-Augmented Generation: instead of Claude 
# guessing from memory, it's handed the actual relevant 
# passages from your PDF before answering.
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever_obj = vectordb.as_retriever()
    return retriever_obj


# model  — which Claude model answers questions
# temperature  — 0 = focused/factual, 1 = creative. 0.5 balances the two, good for QA
# max_tokens  — caps how long an answer can be
# api_key  — pulled from the .env file via os.environ, never hardcoded
def get_llm():
    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        temperature=0.5,
        max_tokens=256,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    return llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# RetrievalQA  wires the retriever and the LLM together into one chain:
# 1. retriever_obj  finds the chunks most relevant to the question
# 2. chain_type="stuff"  “stuffs” all those chunks directly into the prompt sent to Claude
# 3. qa.invoke(query)  runs the chain and response['result']  is Claude's final written answer

def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
 
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
 
    chain = (
        {"context": retriever_obj | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
 
    return chain.invoke(query)
 



rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Upload PDF File",
                file_count="single",
                file_types=[".pdf"],
                type="filepath"),
        gr.Textbox(label="Input Query", lines=2,
                   placeholder="Type your question..."),
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot (Claude-powered)",
)


rag_application.launch()

#Note: to run the app ( uv run qabot.py)