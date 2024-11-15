from langchain.embeddings import HuggingFaceBgeEmbeddings

from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from langchain.schema import Document
import pandas as pd

from langchain_core.documents import Document


def load_eval_file(filename):
    df = pd.read_csv(filename)
    char_to_chunks = {}

    for _, row in df.iterrows():
        character = row['character']
        chunk_ids = row['chunk_id']

        if isinstance(chunk_ids, str):
            chunk_id_list = [int(id_str) for id_str in chunk_ids.split()]
        else:
            chunk_id_list = [int(chunk_ids)]

        if character not in char_to_chunks:
            char_to_chunks[character] = chunk_id_list

    return char_to_chunks

class ChatBotEval:
    load_dotenv()
    def __init__(self):



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

        index_name = "langchain-demo"
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



    def generate_response(self,input):
        result = self.rag_chain.invoke(input)
        return result



@dataclass
class RetrievalMetrics:
    top_k_accuracy: Dict[str, float]
    details: Dict[str, List[str]]  # 存储每个查询的详细信息

class RAGChunkEvaluator:
    def __init__(self, eval_file_path: str, k_values: List[int] = [1, 3, 5]):
        """
        初始化RAG评估器

        Args:
            eval_file_path: 评估文件路径
            k_values: 要计算的top-k值列表
        """
        self.eval_dict = self.load_eval_file(eval_file_path)
        self.k_values = k_values

    def load_eval_file(self, filename: str) -> Dict[str, List[int]]:
        df = pd.read_csv(filename)

        char_to_chunks = {}

        for _, row in df.iterrows():
            character = row['character']
            chunk_ids = row['chunk_id']

            if isinstance(chunk_ids, str):
                chunk_id_list = [int(id_str) for id_str in chunk_ids.split()]
            else:
                chunk_id_list = [int(chunk_ids)]

            if character not in char_to_chunks:
                char_to_chunks[character] = chunk_id_list

        return char_to_chunks

    def _extract_chunk_ids(self, source_documents: List[Document]) -> List[int]:
        """
        从source documents中提取line numbers作为chunk ids

        Args:
            source_documents: RAG检索返回的文档列表

        Returns:
            line numbers列表
        """
        line_numbers = []
        for doc in source_documents:
            line_number = doc.metadata.get('line_number')
            if line_number is not None:
                line_number = int(line_number)
                line_numbers.append(line_number)
        return line_numbers

    def evaluate_single_query(self,
                              query: str,
                              source_documents: List[Document]) -> Dict[str, Any]:
        """
        评估单个查询的检索结果

        Args:
            query: 查询的汉字
            source_documents: RAG检索返回的文档列表

        Returns:
            包含评估结果的字典
        """
        # 获取预测的line numbers（作为chunk ids）
        predicted_chunks = self._extract_chunk_ids(source_documents)

        relevant_chunks = self.eval_dict.get(query, [])

        if not relevant_chunks:
            return {
                "found_chunks": predicted_chunks,
                "relevant_chunks": [],
                "hits": {k: 0 for k in self.k_values},
                "retrieved_contents": [doc.page_content for doc in source_documents]  # 添加检索内容
            }

        # 计算每个k值的命中情况
        hits = {}
        for k in self.k_values:
            top_k_chunks = predicted_chunks[:k]
            # 检查是否有交集
            hit = any(chunk in relevant_chunks for chunk in top_k_chunks)
            hits[k] = 1 if hit else 0

        return {
            "found_chunks": predicted_chunks,
            "relevant_chunks": relevant_chunks,
            "hits": hits,
            "retrieved_contents": [doc.page_content for doc in source_documents]  # 添加检索内容
        }

    def evaluate_batch(self, results: List[Dict[str, Any]]) -> RetrievalMetrics:
        """
        评估一批查询的检索结果

        Args:
            results: 包含查询和检索结果的列表
                每个元素应该是一个字典，包含:
                - query: 查询的汉字
                - source_documents: 检索到的文档列表

        Returns:
            评估指标对象
        """
        all_hits = {k: [] for k in self.k_values}
        details = []

        for result in results:
            query = result['query']
            source_documents = result['source_documents']

            eval_result = self.evaluate_single_query(query, source_documents)


            for k, hit in eval_result['hits'].items():
                all_hits[k].append(hit)

            details.append({
                'query': query,
                'found_chunks': eval_result['found_chunks'],
                'relevant_chunks': eval_result['relevant_chunks'],
                'hits': eval_result['hits'],
                'retrieved_contents': eval_result['retrieved_contents']
            })

        # 计算top-k准确率
        top_k_accuracy = {
            f'top_{k}_accuracy': np.mean(hits)
            for k, hits in all_hits.items()
        }

        return RetrievalMetrics(
            top_k_accuracy=top_k_accuracy,
            details=details
        )

def save_evaluation_results(metrics: RetrievalMetrics, output_csv: str):
    overall_results = {
        'total_queries': len(metrics.details)
    }
    for k, accuracy in metrics.top_k_accuracy.items():
        overall_results[k] = accuracy

    results_df = pd.DataFrame([overall_results])
    results_df.to_csv(output_csv, index=False)
    print(f"\nOverall results saved to {output_csv}")

    return results_df

def evaluate_rag_system(eval_file_path: str, chatbot: Any, output_csv: str):
    """
    评估RAG系统

    Args:
        eval_file_path: 评估文件路径
        chatbot: ChatBot实例
    """
    evaluator = RAGChunkEvaluator(eval_file_path)

    test_questions = list(evaluator.eval_dict.keys())

    results = []
    for character in test_questions:
        question = f"{character} means?"

        answer = chatbot.generate_response(question)
        print("answer", answer)

        results.append({
            'query': character,  # 存储汉字而不是完整的问题
            'source_documents': answer['source_documents']
        })

    metrics = evaluator.evaluate_batch(results)

    results_df = save_evaluation_results(metrics, output_csv)

    print("\nEvaluation Results:")
    print("Top-K Accuracy:", metrics.top_k_accuracy)
    print("\nDetailed Results:")
    for detail in metrics.details:
        print(f"\nCharacter: {detail['query']}")
        print(f"Found chunks (line numbers): {detail['found_chunks']}")
        print(f"Relevant chunks: {detail['relevant_chunks']}")
        print(f"Hits: {detail['hits']}")
        print("Retrieved contents:")
        for content in detail['retrieved_contents']:
            print(f"  - {content}")

    return metrics, results_df

# 使用示例
if __name__ == "__main__":
    # 评估文件路径
    eval_file_path = "./materials/Chinese_with_chunk_label.csv"
    output_csv = "./results/rag_evaluation_overall.csv"

    # 创建chatbot实例
    bot = ChatBotEval()

    # 进行评估
    metrics, results_df = evaluate_rag_system(eval_file_path, bot, output_csv)





