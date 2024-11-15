from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
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

def split_document_by_newline(documents):
    split_docs = []

    for doc in documents:
        lines = doc.page_content.split('\n')

        for line_num, line in enumerate(lines, 1):
            if line.strip():
                split_docs.append(
                    Document(
                        page_content=line,
                        metadata={
                            **doc.metadata,           # 保留原始文檔的metadata
                            'line_number': line_num,  # 添加行號
                            'total_lines': len(lines) # 總行數
                        }
                    )
                )

    return split_docs

class ChatBot:
    load_dotenv()
    def __init__(self):

        print("File Loading...")
        # Load and split documents
        loader = TextLoader('./materials/chinese.txt')
        documents = loader.load()

        print("Chunking...")
        # Chunking
        docs = split_document_by_newline(documents)
        #print(docs)

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
        You are a museum assistant specializing in ancient Chinese characters. Your task is to answer user questions based on the contents of the oracle tool. Follow these steps to complete the task:

        1. Use the provided context to understand the meanings and examples of the Chinese character in question.
        2. Provide the answer in both Chinese and English translation.
        3. If you don't know the answer, just state that you don't know and say sorry.
        4. Ensure your answer is precise and avoid examples that contain gender discrimination or racism.
        5. Do not include any XML tags in your output.

        Use the following format for your response:
        - Meanings: List the meanings of the character in Chinese and English. Provide corresponding examples in Chinese with English translations.

        Context: {context}
        Question: {question}

        Answer:
        Meanings:
        1. [Meaning in Chinese] ([Meaning in English]) - [Example in Chinese]. ([Example in English])
        2. [Meaning in Chinese] ([Meaning in English]) - [Example in Chinese]. ([Example in English])

        You should give the answer just like the following examples:
        Question: What does 爱 mean?
        Answer:
        Meanings: 
        1. 喜爱，爱好 (Like, hobby). [爱]好人物，善诱无倦，士类以此高之。(He likes people and is tireless in his efforts to persuade them, and scholars admire him for this.)
        2. 爱护，加惠，钦慕，爱戴 (Love, favor, admire, love). [爱]此沧江闲白鸥。([Love] The white gulls idle in the vast river.)
        """

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])


        print("Retrieval answer...")
        self.rag_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )


        
# Usage example:
if __name__ == "__main__":
    chatbot = ChatBot()
