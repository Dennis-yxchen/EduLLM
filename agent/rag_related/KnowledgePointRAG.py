import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from collections.abc import Callable
from typing import Iterable
import numpy as np
from rag_related.memory_without_time import NaiveAssociativeMemory

from rag_related.extract_knowledge_point import KnowledgePointExtractor
from rag_related.abstract_rag import AbstractRAG
from memory_without_time import NaiveAssociativeMemory
from collections import abc
from collections.abc import Callable, Iterable, Sequence
import datetime
import json
import threading

import numpy as np
import pandas as pd
from rag_related.importance_model import ConstantImportanceModel

class KnowledgePointMemory(NaiveAssociativeMemory):
    def __init__(self,
                 sentence_embedder : Callable[[str], np.ndarray],
                 importance_threshold: float,
                 max_memories: int,
                 contextualize_size: int,
                 deduplication_threshold: float,
                 importance: Callable[[str, Sequence[tuple[str, float]]], float] | None = None,
                 seed: int | None = None,
                 num_knowledge_points: int = 3,
                 ):
        """构造函数

        参数:
          sentence_embedder: 文本嵌入模型
          importance: 将句子映射到 [0, 1] 范围内的重要性模型，如果为None，则使用一个将所有记忆设置为1.0重要性的恒定重要性模型
          seed: 随机数生成器使用的可选种子。如果为None，则使用默认rng。
        """
        self._memory_bank_lock = threading.Lock()
        self._seed = seed
        self._embedder = sentence_embedder
        self._num_to_retrieve_to_contextualize_importance = contextualize_size
        self._importance = (
            importance or ConstantImportanceModel().importance)

        #  hyperparameters
        self._importance_threshold = importance_threshold
        self._max_memories = max_memories
        self._contextualize_size = contextualize_size
        self._deduplication_threshold = deduplication_threshold
        self._num_knowledge_points = num_knowledge_points
        self._memory_bank = pd.DataFrame(
            columns=['text', 'tags', 'embedding', 'importance'] + 
            ['knowledge_point_' + str(i)  for i in range(self._num_knowledge_points)] +
            ['knowledge_point_' + str(i) + '_embedding' for i in range(self._num_knowledge_points)]
            )
        self._stored_hashes = set()

    def get_state(self) -> dict:
        """将 NaiveAssociativeMemory 转换为字典"""
        with self._memory_bank_lock:
            output = {
                'seed': self._seed,
                'stored_hashes': list(self._stored_hashes),
                'memory_bank': self._memory_bank.to_json(),
            }
        return output
    
    def set_state(self, state: dict) -> None:
        """从字典设置 NaiveAssociativeMemory"""
        with self._memory_bank_lock:
            self._seed = state['seed']
            self._stored_hashes = set(state['stored_hashes'])
            self._memory_bank = pd.read_json(state['memory_bank'])
    
    def add(
        self,
        text: str,
        knowledge_points: Sequence[str], # like [(knowledge_point), (knowledge_point)]
        tags: Iterable[str] = (),
        importance: float | None = None,
    ) -> None:
        """添加非重复条目（文本、标签、重要性）到记忆中

        参数:
          text: 添加到记忆中的内容
          tags: 可选标签
          importance: 可选地设置记忆的重要性。
        """
        if importance is None:
            with self._memory_bank_lock:
                memory_size = len(self._memory_bank)
            num_to_retrieve = self._num_to_retrieve_to_contextualize_importance
            if memory_size < num_to_retrieve:
                num_to_retrieve = memory_size
            context = self.retrieve_random_with_importance(k=num_to_retrieve)
            importance = self._importance(text, context)

        # 移除记忆中的所有换行符。
        text = text.replace('\n', ' ')

        contents = {
            'text': text,
            'tags': tuple(tags),
            'importance': importance,
        }
        hashed_contents = hash(tuple(contents.values()))
        
        for i in range(self._num_knowledge_points):
            if i < len(knowledge_points):
                contents['knowledge_point_' + str(i)] = knowledge_points[i]
            else:
                contents['knowledge_point_' + str(i)] = None
        
        
        
        derived = {'text_embedding': self._embedder(text)}
        for i in range(self._num_knowledge_points):
            if i < len(knowledge_points):
                derived['knowledge_point_' + str(i) + '_embedding'] = self._embedder(knowledge_points[i])
            else:
                derived['knowledge_point_' + str(i) + '_embedding'] = np.zeros(derived['text_embedding'].shape)
        new_df = pd.Series(contents | derived).to_frame().T.infer_objects()

        with self._memory_bank_lock:
            if hashed_contents in self._stored_hashes:
                return
            self._memory_bank = pd.concat(
                [self._memory_bank, new_df], ignore_index=True
            )
            self._stored_hashes.add(hashed_contents)
    
    def extend(
        self,
        texts: Iterable[str],
        **kwargs,
    ) -> None:
        """添加文本到记忆中

        参数:
          texts: 要添加到记忆中的字符串列表
          **kwargs: 传递给 .add 的参数
        """
        for text in texts:
            self.add(text, **kwargs)
    
    def get_data_frame(self) -> pd.DataFrame:
        with self._memory_bank_lock:
            return self._memory_bank.copy()

    def _get_top_k_cosine_base_on_text(self, x: np.ndarray, k: int):
        """返回与输入向量x最相似的前k行

        参数:
          x: 输入向量。
          k: 返回的行数。

        返回:
          按余弦相似度降序排序的行。
        """
        with self._memory_bank_lock:
            cosine_similarities = self._memory_bank['embedding'].apply(
                lambda y: np.dot(x, y)
            )

            # 按余弦相似度降序排序。
            cosine_similarities.sort_values(ascending=False, inplace=True)

            # 返回前k行。
            return self._memory_bank.iloc[cosine_similarities.head(k).index]
        
    def _get_top_k_similar_rows(
        self, x, k: int, use_importance: bool = True
    ):
        """返回与输入向量x最相似的前k行

        参数:
          x: 输入向量。
          k: 返回的行数。
          use_importance: 如果为true则按重要性加权相似度

        返回:
          按相似度降序排序的行。
        """
        with self._memory_bank_lock:
            cosine_similarities = self._memory_bank['embedding'].apply(
                lambda y: np.dot(x, y)
            )

            similarity_score = cosine_similarities

            if use_importance:
                importance = self._memory_bank['importance']
                similarity_score += importance

            # 按相似度降序排序。
            similarity_score.sort_values(ascending=False, inplace=True)

            # 返回前k行。
            return self._memory_bank.iloc[similarity_score.head(k).index]
        
    def _get_similar_rows_with_threshold(
        self, x, threshold: float
    ):
        """
        返回与输入向量x相似度大于阈值的行
        """
        with self._memory_bank_lock:
            cosine_similarities = self._memory_bank['embedding'].apply(
                lambda y: np.dot(x, y)
            )

            similarity_score = cosine_similarities

            # 按相似度降序排序。
            similarity_score.sort_values(ascending=False, inplace=True)

            # 返回前k行。
            return self._memory_bank.iloc[similarity_score[similarity_score > threshold].index]
    
    def _get_top_k_cosine_base_on_keywords(self, keywords: Sequence[str], k: int):
        print(f"keywords: {keywords}")
        input_emb = np.stack([self._embedder(keyword) for keyword in keywords])
        if self._num_knowledge_points > len(keywords):
            input_emb = np.concatenate([input_emb, np.zeros((self._num_knowledge_points - len(keywords), input_emb.shape[1]))])
                
        print(f"input_emb: {input_emb.shape}")
        with self._memory_bank_lock:
            embeddings = []
            for i in range(self._num_knowledge_points):
                # 提取每个knowledge point的列数据（4个样本的768维向量）
                col_data = self._memory_bank[f'knowledge_point_{i}_embedding'].values
                # 将列数据转换为二维数组 (4, 768)
                embeddings.append(np.stack(col_data))

            # 将列表中的三个 (4, 768) 数组合并为 (4, 3, 768)
            db_embeddings = np.stack(embeddings, axis=1)
            print(f"input_emb shape: {input_emb.shape}")
            print(f"db_embeddings shape: {db_embeddings.shape}")
            dot_products = np.einsum('m d, n m d -> n m', input_emb, db_embeddings)
            print(f"dot_products: {dot_products.shape}")
            total_scores = np.mean(dot_products, axis=1)
            print(f"total_scores: {total_scores}")
            
            sorted_indices = np.argsort(total_scores)[::-1][:k]
            # 选择前k个索引
            print(f"sorted: {self._memory_bank.iloc[sorted_indices]}")

            return self._memory_bank.iloc[sorted_indices]
    
    def _pd_to_text(
        self,
        data: pd.DataFrame,
    ) -> Sequence[str]:
        """将数据框格式化为字符串列表

        参数:
          data: 要处理的数据框

        返回:
          字符串列表，每个记忆对应一个字符串
        """
        output = data['text']
        return output.tolist()
    
    def retrieve_associative_with_text(
            self,
            query: str,
            k: int = 1,
            use_importance: bool = True,
    ) -> Sequence[str]:
        """Retrieve memories using associative retrieval

        Args:
            query: Query string for retrieval
            k: Number of memories to retrieve
            use_importance: Whether to use importance weighting

        Returns:
            List of memory strings
        """
        # Get query embedding
        query_embedding = self._embedder(query)

        # Get similar rows using pandas DataFrame
        similar_rows = self._get_top_k_similar_rows(
            query_embedding,
            k,
            use_importance=use_importance,
        )

        # Convert to text format
        return self._pd_to_text(similar_rows)
    
    
    def retrieve_associative_with_keywords(
            self,
            keywords: Sequence[str],
            k: int = 3,
    ) -> Sequence[str]:
        
        similar_rows = self._get_top_k_cosine_base_on_keywords(
            keywords,
            k,
        )
        
        # Convert to text format
        return self._pd_to_text(similar_rows)
    
    def retrieve_by_similarity_with_threshold_by_text(
        self,
        query: str,
        threshold: float = 0.7,
    ) -> Sequence[str]:
        """关联检索记忆

        参数:
          query: 用于检索的字符串
          k: 要检索的记忆数量
          use_importance: 是否使用重要性进行检索

        返回:
          对应记忆的字符串列表
        """
        query_embedding = self._embedder(query)

        data = self._get_similar_rows_with_threshold(
            query_embedding,
            threshold
        )

        return self._pd_to_text(data)

    def retrieve_by_regex(
        self,
        regex: str,
    ) -> Sequence[str]:
        """通过正则表达式检索记忆

        参数:
          regex: 匹配的正则表达式

        返回:
          对应记忆的字符串列表
        """
        with self._memory_bank_lock:
            data = self._memory_bank[self._memory_bank['text'].str.contains(regex)]

        return self._pd_to_text(data)

    def retrieve_random(
        self,
        k: int = 1,
    ) -> Sequence[str]:
        """检索随机的记忆

        参数:
          k: 要检索的记忆数量

        返回:
          对应记忆的字符串列表
        """
        with self._memory_bank_lock:
            data = self._memory_bank.sample(k, random_state=self._seed)
        return self._pd_to_text(data)

    def retrieve_random_with_importance(
        self,
        k: int = 1,
    ) -> Sequence[tuple[str, float]]:
        """检索随机的记忆并返回重要性

        参数:
          k: 要检索的记忆数量

        返回:
          对应记忆的字符串列表及其重要性值
        """
        with self._memory_bank_lock:
            data = self._memory_bank.sample(k, random_state=self._seed)
        return tuple(zip(list(data['text']), list(data['importance'])))

    def __len__(self):
        """返回记忆库中的条目数量

        由于记忆不能被删除，长度不会减少，可用于检查记忆库的内容是否发生变化。
        """
        with self._memory_bank_lock:
            return len(self._memory_bank)

    def get_mean_importance(self) -> float:
        """返回记忆库中记忆的重要性的平均值"""
        with self._memory_bank_lock:
            return self._memory_bank['importance'].mean()

    def get_max_importance(self) -> float:
        """返回记忆库中记忆的最大重要性"""
        with self._memory_bank_lock:
            return self._memory_bank['importance'].max()

    def get_min_importance(self) -> float:
        """返回记忆库中记忆的重要性的最小重要性"""
        with self._memory_bank_lock:
            return self._memory_bank['importance'].min()

    def set_num_to_retrieve_to_contextualize_importance(
        self, num_to_retrieve: int) -> None:
        """设置用于上下文化重要性的要检索的记忆数量

        设置为0以禁用重要性的上下文化

        参数:
          num_to_retrieve: 用于上下文化重要性的要检索的记忆数量
        """
        self._num_to_retrieve_to_contextualize_importance = num_to_retrieve

    def get_all_memories_as_text(self) -> Sequence[str]:
        """返回记忆库中的所有记忆作为字符串序列"""
        memories_data_frame = self.get_data_frame()
        texts = self._pd_to_text(memories_data_frame)
        return texts
    
    def save(self, file_path: str) -> None:
        """将记忆库保存到 JSON 文件"""
        state = self.get_state()
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)

    def load(self, file_path: str) -> None:
        """从 JSON 文件加载记忆库"""
        with open(file_path, 'r') as f:
            state = json.load(f)
        # 将 embedding 从列表转换回 np.ndarray
        for item in state['memory_bank']:
            item['embedding'] = np.array(item['embedding'])
        self.set_state(state)
    
class KnowledgePointRAG(AbstractRAG):
    def __init__(self, 
                 model, 
                 sentence_embedder: Callable[[str], np.ndarray],
                 memory_bank: KnowledgePointMemory,
                #  k: int,
                #  similarity_threshold: float,
                #  use_importance_weighting: bool,
                 min_similarity_score: float):
        super().__init__(model, sentence_embedder, memory_bank)
        self._model = model
        self._embedder = sentence_embedder
        self._memory_bank = memory_bank
        self._stored_hashes = set()
        # self._num_of_question_to_retrieve = k
        # self._similarity_threshold = similarity_threshold
        # self._use_importance = use_importance_weighting
        self._min_similarity_score = min_similarity_score
            
    def retrieve_question_by_similarity(self, question, num_of_question_to_retrieve):
        return self._memory_bank.retrieve_associative_with_text(query = question,
                                                        k = num_of_question_to_retrieve)
        
    def retrieve_question_by_threshold(self, question, threshold):
        return self._memory_bank.retrieve_by_similarity_with_threshold_by_text(query = question,
                                                        threshold = threshold)
    
    def retrieve_question_by_keywords(self, keywords, num_of_question_to_retrieve):
        return self._memory_bank.retrieve_associative_with_keywords(keywords = keywords,
                                                        k = num_of_question_to_retrieve)
    
    def add_question_to_memory(self, 
                               question:str,
                               knowledge_points: Sequence[str],
                               tags:Iterable[str] = ()):
        self._memory_bank.add(text = question, knowledge_points = knowledge_points,tags = tags)
        
        

if __name__ == "__main__":
    # Example usage
    import sentence_transformers
    st_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2')
    embedder = lambda x: st_model.encode(x, show_progress_bar=False)

    memory = KnowledgePointMemory(
        sentence_embedder=embedder,
        importance_threshold=0.5,
        max_memories=1000,
        contextualize_size=5,
        deduplication_threshold=0.1,
        num_knowledge_points=3
    )

    rag = KnowledgePointRAG(
        model=None,  # Replace with your actual model
        sentence_embedder=embedder,
        memory_bank=memory,
        min_similarity_score=0.7
    )

    rag.add_question_to_memory("Reverse a string in Python", knowledge_points=["string slicing", "loop structures", "function returns"])
    rag.add_question_to_memory("Count element frequency in a list", knowledge_points=["dictionary manipulation", "list iteration", "conditional updates"])
    rag.add_question_to_memory("Generate Fibonacci sequence", knowledge_points=["recursion/iteration", "list appending", "loop control"])
    rag.add_question_to_memory("Count word frequencies in a text file", knowledge_points=["file I/O", "string splitting", "dictionary counting"])
    rag.add_question_to_memory("Create a function timing decorator", knowledge_points=["decorator syntax", "time module", "closures"])
    rag.add_question_to_memory("Implement Student subclass inheritance", knowledge_points=["class inheritance", "super() method", "attribute encapsulation"])
    rag.add_question_to_memory("Remove list duplicates while preserving order", knowledge_points=["set operations", "list traversal", "order preservation"])
    rag.add_question_to_memory("Build a context manager for file handling", knowledge_points=["__enter__/__exit__", "exception handling", "resource management"])
    rag.add_question_to_memory("Create custom AgeError exception", knowledge_points=["exception inheritance", "raise statements", "try-except blocks"])
    rag.add_question_to_memory("Dynamically import modules using strings", knowledge_points=["importlib", "getattr()", "dynamic execution"])
    rag.add_question_to_memory("Calculate portfolio risk using CAPM", 
           knowledge_points=["capital_asset_pricing_model", "beta_calculation", "market_risk_premium"])

    rag.add_question_to_memory("Optimize investment allocation with Markowitz model", 
            knowledge_points=["efficient_frontier", "covariance_matrix", "risk_return_tradeoff"])

    rag.add_question_to_memory("Evaluate corporate capital structure using WACC", 
            knowledge_points=["weighted_average_cost_of_capital", "debt_equity_ratio", "cost_of_capital_components"])

    rag.add_question_to_memory("Simulate Value-at-Risk (VaR) for stock holdings", 
            knowledge_points=["historical_simulation", "monte_carlo_methods", "confidence_levels"])

    rag.add_question_to_memory("Analyze financial health via DuPont ROE decomposition", 
            knowledge_points=["return_on_equity", "profit_margins", "asset_turnover_ratio"])
    print(rag.retrieve_question_by_keywords(['coding'], 4))