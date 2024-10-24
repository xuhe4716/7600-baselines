from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
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


class ChatBot:
    load_dotenv()
    def __init__(self):

        # Load and split documents
        loader = TextLoader('./materials/Rag_document.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=40)
        docs = text_splitter.split_documents(documents)


        # Initialize embeddings
        model_name = "BAAI/bge-base-zh-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
            #query_instruction="为这个句子生成表示以用于检索相关文章："
        )
        #embeddings.query_instruction = "为这个句子生成表示以用于检索相关文章："
        #embeddings = HuggingFaceEmbeddings()


        # Initialize Pinecone instance
        pc = Pinecone(api_key= os.getenv('PINECONE_API_KEY'))

        index_name = "langchain-demo"

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )            
            )
        index = pc.Index(index_name)
        docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

        # Initialize ChatOpenAI
        model_name = "gpt-3.5-turbo"
        llm = ChatOpenAI(model_name=model_name, organization='')


        # Define prompt template
        template = """
        You are a museum assistant for ancient Chinese characters. Users will ask you questions about Chinese characters. Use the following piece of context to answer the question.
        You should tell the user the 3 义项描述 of the word, and corresponding 语料 examples of the character.
        If you don't know the answer, just say you don't know.
        Your answer should be concise.

        Context: {context}
        Question: {question}
        Answer:
        """

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        self.rag_chain = RetrievalQA.from_chain_type(
            llm, retriever=docsearch.as_retriever(), chain_type_kwargs={"prompt": prompt}
        )

        
# Usage example:
if __name__ == "__main__":
    chatbot = ChatBot()
