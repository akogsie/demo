from fastapi import FastAPI, Form, Request
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional,Any,Generator


from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import QAGenerationChain

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


text_splitter = RecursiveCharacterTextSplitter()



embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device':'cpu'})


db= FAISS.load_local('./vectorstore/db_faiss',embeddings,allow_dangerous_deserialization=True)


llm = Ollama(model="gemma:7b")


qa_template = """You are a smart chat bot form answering user questions. Use the following pieces of information to answer the user's question. If you don't know the answer, just say you don't know, don't try to make up an answer.\
please provide response in neat bullet points that is readbable to the user
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful Answer:
"""



prompt = PromptTemplate(template=qa_template, input_variables=['context','question'])


qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                 retriever=db.as_retriever(search_kwargs={'k':5}),
                                                          return_source_documents=True,
                                                          chain_type_kwargs={'prompt':prompt})


### Create FASTAPI Instance
app = FastAPI()
templates = Jinja2Templates(directory="templates/")

app.mount("/static",StaticFiles(directory="static"), name="static")


def bot(user_input):
    search_query = str(user_input)
    response = qa.invoke(search_query)
    ai_response = response['result']
    return ai_response


@app.get("/qa_bot",response_class=HTMLResponse)
async def index_page(request:Request, user_input:str =Form(None)):
    user_input = user_input
    return templates.TemplateResponse('index.html', context={'request':request, 'user_input': user_input})


@app.post('/cbn_chat_bot')
def submit(request:Request, user_input: str = Form(None)):
    result = bot(user_input)
    return templates.TemplateResponse('index.html', context={'request':request, 'user_input': user_input, 'result': result})


if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)

    