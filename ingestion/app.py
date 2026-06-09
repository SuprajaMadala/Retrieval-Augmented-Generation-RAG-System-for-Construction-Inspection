from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, Runnable
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector 
from langchain_community.llms import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TextSplitter
from sqlalchemy import create_engine
from icecream import ic # type: ignore
from flask import Flask,request,jsonify


embeds= OllamaEmbeddings(model="mistral")
eng=create_engine("postgresql://supraja:asdf@localhost:5433/vector_db")
vectorstore =PGVector(
    connection=eng,
    embeddings=embeds,
    collection_name="pdf_data"
)
llm_model = OllamaLLM(model="mistral")
ic("Connection Successfull")



class QueryLLM:
  

    def query_bedrock(self, question: str) -> str:
        retriever = vectorstore.as_retriever()
        template = """
        Question: {question},
        Context: {context}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
            | prompt
            | llm_model
            | StrOutputParser()
        ).with_types(input_type=str)
        result = chain.invoke(question)
        return result

query_llm = QueryLLM()
app = Flask(__name__)
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    try:
        answer = query_llm.query_bedrock(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
