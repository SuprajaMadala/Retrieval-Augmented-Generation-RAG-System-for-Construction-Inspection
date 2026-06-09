from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, Runnable
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector  # type: ignore
from langchain_community.llms import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TextSplitter
from sqlalchemy import create_engine
from icecream import ic # type: ignore

embeds= OllamaEmbeddings(model="mistral")
eng=create_engine("postgresql://supraja:asdf@localhost:5433/vector_db")
vectorstore =PGVector(
    connection=eng,
    embeddings=embeds,
    collection_name="pdf_data"
)
llm_model = OllamaLLM(model="mistral")
ic("Connection Successfull")



# -------> Load the pdf

pdf = PyPDFLoader(file_path="USDOT - Superpave Asphalt Mixture Design.pdf")

pdf_data= pdf.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=700 ,chunk_overlap=300))

import time
pdf_text_data=[]
for page in pdf_data:
    pdf_text_data.append(page.page_content)
    # ic(page.page_content)
ic(len(pdf_text_data))
i=0
try:
    for data in pdf_text_data:
        ic(i)
        i=i+1
        print(data)
        vectorstore.add_texts([data])

    ic("Successfully embedded and stored in vector database")
except Exception as e:
    ic(e)




