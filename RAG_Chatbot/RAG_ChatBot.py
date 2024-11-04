from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_pinecone import PineconeVectorStore
import ollama
from openai import OpenAI
from langchain_openai import ChatOpenAI
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4

from langchain_core.documents import Document

class ChatBot:
    load_dotenv()
    def __init__(self):

        print("File Loading...")
        # Load and split documents
        loader = TextLoader('./materials/chinese.txt')
        documents = loader.load()

        print("Chunking...")
        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        docs = text_splitter.split_documents(documents)

        print("Initialize embeddings...")
        # Initialize embeddings
        model_name = "BAAI/bge-base-zh-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        print("Initialize vector database...")
        # Initialize Pinecone instance
        pc = Pinecone(api_key= os.getenv('PINECONE_API_KEY'))

        index_name = "langchain-demo"

        if index_name not in pc.list_indexes().names():
            print("yes")
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            #index = pc.Index(index_name)
            docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)


        print("Initialize ChatOpenAI...")
        # Initialize ChatOpenAI
        model_name = "gpt-4o-mini"
        llm = ChatOpenAI(model_name=model_name, temperature=0)


        # Define prompt template
        template = """
        You are a museum assistant for ancient Chinese characters. Users will ask you questions about Chinese characters. Use the following piece of context to answer the question.
        You should give the meanings and corresponding examples.
        You should give the answer in English. Just like the following examples:
        Question: What does 爱 mean?
        Answer: Meanings: \
                1.Like, hobby (喜爱，爱好)\
                2.love, favor, admire, love (爱护，加惠，钦慕，爱戴)\
                Examples: \
                1.[爱]此沧江闲白鸥。([Love] The white gulls idle in the vast river.)\
                2.[爱]好人物，善诱无倦，士类以此高之。(He likes people and is tireless in his efforts to persuade them, and scholars admire him for this.)
        If you don't know the answer, just say sorry to the user and say you don't know.
        Your answer should be precise.
        
        Context: {context}
        Question: {question}
        Answer:
        """

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])


        print("Retrieval answer...")
        self.rag_chain = RetrievalQA.from_chain_type(
            llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), chain_type_kwargs={"prompt": prompt}
        )


        
# Usage example:
if __name__ == "__main__":
    chatbot = ChatBot()
