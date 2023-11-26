# gradio
import gradio as gr
import random
import time
#boto3 for S3 access
import boto3
from botocore import UNSIGNED
from botocore.client import Config
# access .env file
from dotenv import load_dotenv
#from bs4 import BeautifulSoup
# HF libraries
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
# vectorestore
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
# retrieval chain
from langchain.chains import RetrievalQA
# prompt template
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
# logging
#import logging
import zipfile

# load HF Token
config = load_dotenv(".env")


model_id = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={
    "temperature":0.1, 
    "max_new_tokens":1024, 
    "repetition_penalty":1.2, 
    "streaming": True, 
    "return_full_text":True
    })

model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
embeddings = HuggingFaceHubEmbeddings(repo_id=model_name)


s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

## Chroma DB
s3.download_file('rad-rag-demos', 'vectorstores/chroma.sqlite3', './chroma_db/chroma.sqlite3')
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
db.get()

## FAISS DB
# s3.download_file('rad-rag-demos', 'vectorstores/faiss_db_ray.zip', './chroma_db/faiss_db_ray.zip')
# with zipfile.ZipFile('./chroma_db/faiss_db_ray.zip', 'r') as zip_ref:
#     zip_ref.extractall('./chroma_db/')

# FAISS_INDEX_PATH='./chroma_db/faiss_db_ray'
# db = FAISS.load_local(FAISS_INDEX_PATH, embeddings)

retriever = db.as_retriever(search_type = "mmr")#, search_kwargs={'k': 5, 'fetch_k': 25})
global qa 
template = """
You are the friendly documentation buddy Arti, who helps the Human in using RAY, the open-source unified framework for scaling AI and Python applications.\
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question :
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)
memory = ConversationBufferMemory(memory_key="history", input_key="question")
qa = RetrievalQA.from_chain_type(llm=model_id, chain_type="stuff", retriever=retriever, verbose=True, return_source_documents=True, chain_type_kwargs={
    "verbose": True,
    "memory": memory,
    "prompt": prompt
}
    )


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0], history)
    print(*memory)
    sources = [doc.metadata.get("source") for doc in response['source_documents']]
    src_list = '\n'.join(sources)
    print_this = response['result']+"\n\n\n Sources: \n\n\n"+src_list

    #history[-1][1] = ""
    #for character in response['result']: #print_this:
    #    history[-1][1] += character
    #    time.sleep(0.05)
    #    yield history
    history[-1][1] = print_this #response['result']
    return history

def infer(question, history):
    query =  question
    result = qa({"query": query, "history": history, "question": question})
    return result

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with your Documentation</h1>
    <p style="text-align: center;">Chat with Documentation, <br />
    when everything is ready, you can start asking questions about the docu ;)</p>
</div>
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)      
        chatbot = gr.Chatbot([], elem_id="chatbot")
        clear = gr.Button("Clear")
        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
    question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()