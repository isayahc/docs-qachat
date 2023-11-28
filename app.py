# gradio
import gradio as gr
#import random
#import time
#boto3 for S3 access
import boto3
from botocore import UNSIGNED
from botocore.client import Config
# access .env file
import os
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
from langchain.chains import RetrievalQAWithSourcesChain
# prompt template
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
# logging
import logging
import zipfile
#contextual retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
# streaming
#from threading import Thread
#from transformers import TextIteratorStreamer


# load .env variables
config = load_dotenv(".env")
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')
AWS_S3_LOCATION=os.getenv('AWS_S3_LOCATION')
AWS_S3_FILE=os.getenv('AWS_S3_FILE')
VS_DESTINATION=os.getenv('VS_DESTINATION')

# initialize Model config
model_id = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={
    "temperature":0.1, 
    "max_new_tokens":1024, 
    "repetition_penalty":1.2, 
    "streaming": True, 
    "return_full_text":True
    })

model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
embeddings = HuggingFaceHubEmbeddings(repo_id=model_name)

# retrieve vectorsrore
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

## Chroma DB
s3.download_file(AWS_S3_LOCATION, AWS_S3_FILE, VS_DESTINATION)
db = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
db.get()

## FAISS DB
# s3.download_file('rad-rag-demos', 'vectorstores/faiss_db_ray.zip', './chroma_db/faiss_db_ray.zip')
# with zipfile.ZipFile('./chroma_db/faiss_db_ray.zip', 'r') as zip_ref:
#     zip_ref.extractall('./chroma_db/')

# FAISS_INDEX_PATH='./chroma_db/faiss_db_ray'
# db = FAISS.load_local(FAISS_INDEX_PATH, embeddings)

retriever = db.as_retriever(search_type = "mmr")#, search_kwargs={'k': 5, 'fetch_k': 25})

compressor = LLMChainExtractor.from_llm(model_id)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
# embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
# compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)

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

# logging for the chain
logging.basicConfig()
logging.getLogger("langchain.chains").setLevel(logging.INFO)    


# qa = RetrievalQA.from_chain_type(llm=model_id, chain_type="stuff", retriever=compression_retriever, verbose=True, return_source_documents=True, chain_type_kwargs={
#     "verbose": True,
#     "memory": memory,
#     "prompt": prompt
# }
#     )
qa = RetrievalQAWithSourcesChain.from_chain_type(llm=model_id, retriever=compression_retriever, verbose=True, chain_type_kwargs={
    "verbose": True,
    "memory": memory,
    "prompt": prompt,
    "document_variable_name": "context"
}
    )

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0], history)
    print(*response)
    print(*memory)
    sources = [doc.metadata.get("source") for doc in response['sources']]
    src_list = '\n'.join(sources)
    print_this = response['answer'] + "\n\n\n Sources: \n\n\n" + src_list
    #sources = f"`Sources:`\n\n' + response['sources']"

    #history[-1][1] = ""
    #for character in response['result']: #print_this:
    #    history[-1][1] += character
    #    time.sleep(0.05)
    #    yield history
    history[-1][1] = response['answer']
    return history #, sources

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

demo.queue().launch()