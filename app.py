
# logging
import logging

# access .env file
import os
from dotenv import load_dotenv

import time

#boto3 for S3 access
import boto3
from botocore import UNSIGNED
from botocore.client import Config

# HF libraries
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
# vectorestore
from langchain.vectorstores import Chroma

# retrieval chain
from langchain.chains import RetrievalQAWithSourcesChain
# prompt template
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import BM25Retriever, EnsembleRetriever
# reorder retrived documents
# github issues
from langchain.document_loaders import GitHubIssuesLoader
# debugging
from langchain.globals import set_verbose
# caching
from langchain.globals import set_llm_cache
# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache

# gradio
import gradio as gr

# template for prompt
from prompt import template



set_verbose(True)

# load .env variables
config = load_dotenv(".env")
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')
AWS_S3_LOCATION=os.getenv('AWS_S3_LOCATION')
AWS_S3_FILE=os.getenv('AWS_S3_FILE')
VS_DESTINATION=os.getenv('VS_DESTINATION')

# initialize Model config
llm_model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# changed named to model_id to llm as is common
llm = HuggingFaceHub(repo_id=llm_model_name, model_kwargs={
    # "temperature":0.1, 
    "max_new_tokens":1024, 
    "repetition_penalty":1.2, 
#    "streaming": True, 
#    "return_full_text":True
    })

# initialize Embedding config
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceHubEmbeddings(repo_id=embedding_model_name)

# remove old vectorstore
if os.path.exists(VS_DESTINATION):
    os.remove(VS_DESTINATION)

# remove old sqlite cache
if os.path.exists('.langchain.sqlite'):
    os.remove('.langchain.sqlite')

set_llm_cache(SQLiteCache(database_path=".langchain.sqlite"))

# retrieve vectorsrore
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

## Chroma DB
s3.download_file(AWS_S3_LOCATION, AWS_S3_FILE, VS_DESTINATION)
# use the cached embeddings instead of embeddings to speed up re-retrival
db = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
db.get()


retriever = db.as_retriever(search_type="mmr")#, search_kwargs={'k': 3, 'lambda_mult': 0.25})

# asks LLM to create 3 alternatives baed on user query


# asks LLM to extract relevant parts from retrieved documents


global qa 

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)
memory = ConversationBufferMemory(memory_key="history", input_key="question")

# logging for the chain
logging.basicConfig()
logging.getLogger("langchain.retrievers").setLevel(logging.INFO)    
logging.getLogger("langchain.chains.qa_with_sources").setLevel(logging.INFO)    




qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True, verbose=True, chain_type_kwargs={
    "verbose": True,
    "memory": memory,
    "prompt": prompt,
    "document_variable_name": "context"
}
    )


#####
#
# Gradio fns
####

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0], history)
    sources = [doc.metadata.get("source") for doc in response['source_documents']]
    src_list = '\n'.join(sources)
    print_this = response['answer'] + "\n\n\n Sources: \n\n\n" + src_list


    history[-1][1] = print_this #response['answer']
    return history

def infer(question, history):
    query =  question
    result = qa({"query": query, "history": history, "question": question})
    return result

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 1920px;">
    <h1>Chat with your Documentation</h1>
    <p style="text-align: center;">This is a privately hosten Docs AI Buddy, <br />
    It will help you with any question regarding the documentation of Ray ;)</p>
</div>
"""



with gr.Blocks(css=css) as demo:
    with gr.Column(min_width=900, elem_id="col-container"):
        gr.HTML(title)      
        chatbot = gr.Chatbot([], elem_id="chatbot")
        #with gr.Row():
        #    clear = gr.Button("Clear")

        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
        with gr.Row():
            clear = gr.ClearButton([chatbot, question])

    question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
        bot, chatbot, chatbot
    )
    #clear.click(lambda: None, None, chatbot, queue=False)

demo.queue().launch()