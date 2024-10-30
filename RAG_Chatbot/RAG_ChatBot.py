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
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


class ChatBot:
    load_dotenv()
    def __init__(self):

        # Load and split documents
        loader = TextLoader('./materials/chinese.txt')
        # loader = TextLoader('./materials/torontoTravelAssistant.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=40)
        docs = text_splitter.split_documents(documents)


        # Initialize embeddings
        model_name = "BAAI/bge-base-zh-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # Initial vector db
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        vector_dir = "vector"
        if not os.path.exists(f'./materials/{vector_dir}'):
            os.makedirs(vector_dir)
            print(f"Created directory: {vector_dir}")

        vector_store.save_local('./materials/vector')

        # Initialize ChatOpenAI
        model_name = "gpt-3.5-turbo"
        llm = ChatOpenAI(model_name=model_name, organization='')


        # Define prompt template
        template = """
        You are a museum assistant for ancient Chinese characters. Users will ask you questions about Chinese characters. Use the following piece of context to answer the question.
        You should tell the user the 3 义项描述 of the word, and corresponding 语料 examples of the character.
        You should give the answer in English. Just like the following examples:
        Question: What does 爱 mean?
        Answer: Meanings: \
                1.Like, hobby (喜爱，爱好)\
                2.love, favor, admire, love (爱护，加惠，钦慕，爱戴)\
                Examples: \
                1.[爱]此沧江闲白鸥。([Love] The white gulls idle in the vast river.)\
                2.[爱]好人物，善诱无倦，士类以此高之。(He likes people and is tireless in his efforts to persuade them, and scholars admire him for this.)
        If you don't know the answer, just say you don't know.
        Your answer should be precise.
        
        Context: {context}
        Question: {question}
        Answer:
        """

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        self.rag_chain = RetrievalQA.from_chain_type(
            llm, retriever=vector_store.as_retriever(), chain_type_kwargs={"prompt": prompt}
        )


        
# Usage example:
if __name__ == "__main__":
    chatbot = ChatBot()
