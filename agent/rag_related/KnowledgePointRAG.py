import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from collections.abc import Callable
from typing import Iterable
import numpy as np
from rag_related.memory_without_time import NaiveAssociativeMemory

from rag_related.extract_knowledge_point import KnowledgePointExtractor
from rag_related.abstract_rag import AbstractRAG
from rag_related.memory_without_time import NaiveAssociativeMemory
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
        """Constructor

        Args:
          sentence_embedder: Text embedding model
          importance: A model that maps sentences to importance values in the range [0, 1]. 
                  If None, a constant importance model is used that sets all memories to an importance of 1.0.
          seed: Optional seed for the random number generator. If None, the default RNG is used.
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
        """Convert NaiveAssociativeMemory to a dictionary"""
        with self._memory_bank_lock:
            output = {
                'seed': self._seed,
                'stored_hashes': list(self._stored_hashes),
                'memory_bank': self._memory_bank.to_json(),
            }
        return output
    
    def set_state(self, state: dict) -> None:
        """Set NaiveAssociativeMemory from a dictionary"""
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
        """Add unique entries (text, tags, importance) to memory.

        Args:
          text: Content to be added to memory.
          tags: Optional tags.
          importance: Optionally set the importance of the memory.
        """
        if importance is None:
            with self._memory_bank_lock:
                memory_size = len(self._memory_bank)
            num_to_retrieve = self._num_to_retrieve_to_contextualize_importance
            if memory_size < num_to_retrieve:
                num_to_retrieve = memory_size
            context = self.retrieve_random_with_importance(k=num_to_retrieve)
            importance = self._importance(text, context)

        # Remove all newline characters from the memory.
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
        """Add texts to memory.

        Args:
          texts: List of strings to add to memory.
          **kwargs: Parameters to pass to the .add method.
        """
        for text in texts:
            self.add(text, **kwargs)
    
    def get_data_frame(self) -> pd.DataFrame:
        with self._memory_bank_lock:
            return self._memory_bank.copy()

    def _get_top_k_cosine_base_on_text(self, x: np.ndarray, k: int):
        """Return the top-k rows most similar to the input vector x

        Args:
          x: Input vector.
          k: Number of rows to return.

        Returns:
          Rows sorted in descending order of cosine similarity.
        """
        with self._memory_bank_lock:
            cosine_similarities = self._memory_bank['text_embedding'].apply(
                lambda y: np.dot(x, y)
            )

            # Sort by cosine similarity in descending order.
            cosine_similarities.sort_values(ascending=False, inplace=True)

            # Return the top-k rows.
            return self._memory_bank.iloc[cosine_similarities.head(k).index]
        
    def _get_top_k_similar_rows(
        self, x, k: int, use_importance: bool = True
    ):
        """Return the top-k rows most similar to the input vector x

        Args:
          x: Input vector.
          k: Number of rows to return.
          use_importance: If true, weight similarity by importance.

        Returns:
          Rows sorted in descending order of similarity.
        """
        with self._memory_bank_lock:
            cosine_similarities = self._memory_bank['text_embedding'].apply(
                lambda y: np.dot(x, y)
            )
            similarity_score = cosine_similarities

            if use_importance:
                importance = self._memory_bank['importance']
                similarity_score += importance

            # Sort by similarity in descending order.
            similarity_score.sort_values(ascending=False, inplace=True)

            # Return the top-k rows.
            return self._memory_bank.iloc[similarity_score.head(k).index]
        
    def _get_similar_rows_with_threshold(
        self, x, threshold: float
    ):
        """
        Returns rows with similarity to input vector x greater than the threshold.
        """
        with self._memory_bank_lock:
            cosine_similarities = self._memory_bank['text_embedding'].apply(
                lambda y: np.dot(x, y)
            )

            similarity_score = cosine_similarities

            # Sort by similarity in descending order.
            similarity_score.sort_values(ascending=False, inplace=True)

            # Return the top k rows.
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
                # Extract column data for each knowledge point
                col_data = self._memory_bank[f'knowledge_point_{i}_embedding'].values
                # Convert column data to 2D arrays (4, 768)
                embeddings.append(np.stack(col_data))

            # Combine the three (4, 768) arrays in the list into a (4, 3, 768) array
            db_embeddings = np.stack(embeddings, axis=1)
            dot_products = np.einsum('m d, n m d -> n m', input_emb, db_embeddings)
            total_scores = np.mean(dot_products, axis=1)
            
            sorted_indices = np.argsort(total_scores)[::-1][:k]

            return self._memory_bank.iloc[sorted_indices]
    
    def _pd_to_text(
        self,
        data: pd.DataFrame,
    ) -> Sequence[str]:
        """Format the DataFrame into a list of strings.

        Args:
          data: The DataFrame to process.

        Returns:
          A list of strings, where each memory corresponds to a string.
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
        print('query_embedding:', query_embedding.shape)

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
    ) -> dict[str: Sequence[str]]:
        
        similar_rows = self._get_top_k_cosine_base_on_keywords(
            keywords,
            k,
        )
        
        # Convert to text format
        return {'text': self._pd_to_text(similar_rows),
                'knowledge_points': similar_rows[[f'knowledge_point_{i}' for i in range(self._num_knowledge_points)]].values.tolist()}
    
    def retrieve_by_similarity_with_threshold_by_text(
        self,
        query: str,
        threshold: float = 0.7,
    ) -> Sequence[str]:
        """Retrieve associative memories

        Args:
          query: String used for retrieval
          k: Number of memories to retrieve
          use_importance: Whether to use importance for retrieval

        Returns:
          List of corresponding memory strings
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
        """Retrieve memories using a regular expression.

        Args:
          regex: The regular expression to match.

        Returns:
          A list of memory strings that match the regex.
        """
        with self._memory_bank_lock:
            data = self._memory_bank[self._memory_bank['text'].str.contains(regex)]

        return self._pd_to_text(data)

    def retrieve_random(
        self,
        k: int = 1,
    ) -> Sequence[str]:
        """Retrieve random memories.

        Args:
          k: Number of memories to retrieve.

        Returns:
          A list of corresponding memory strings.
        """
        with self._memory_bank_lock:
            data = self._memory_bank.sample(k, random_state=self._seed)
        return self._pd_to_text(data)

    def retrieve_random_with_importance(
        self,
        k: int = 1,
    ) -> Sequence[tuple[str, float]]:
        """Retrieve random memories and return their importance.

        Args:
          k: Number of memories to retrieve.

        Returns:
          A list of corresponding memory strings and their importance values.
        """
        with self._memory_bank_lock:
            data = self._memory_bank.sample(k, random_state=self._seed)
        return tuple(zip(list(data['text']), list(data['importance'])))

    def __len__(self):
        """Return the number of entries in the memory bank.

        Since memories cannot be deleted, the length will not decrease, 
        and it can be used to check whether the content of the memory bank has changed.
        """
        with self._memory_bank_lock:
            return len(self._memory_bank)

    def get_mean_importance(self) -> float:
        """Return the average importance of memories in the memory bank."""
        with self._memory_bank_lock:
            return self._memory_bank['importance'].mean()

    def get_max_importance(self) -> float:
        """Return the maximum importance of memories in the memory bank."""
        with self._memory_bank_lock:
            return self._memory_bank['importance'].max()

    def get_min_importance(self) -> float:
        """get min"""
        with self._memory_bank_lock:
            return self._memory_bank['importance'].min()

    def set_num_to_retrieve_to_contextualize_importance(
        self, num_to_retrieve: int) -> None:
        """set num retrieve
        """
        self._num_to_retrieve_to_contextualize_importance = num_to_retrieve

    def get_all_memories_as_text(self) -> Sequence[str]:
        """get string from df"""
        memories_data_frame = self.get_data_frame()
        texts = self._pd_to_text(memories_data_frame)
        return texts
    
    def save(self, file_path: str) -> None:
        """get json"""
        state = self.get_state()
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)

    def load(self, file_path: str) -> None:
        """load json"""
        with open(file_path, 'r') as f:
            state = json.load(f)
        for item in state['memory_bank']:
            item['embedding'] = np.array(item['embedding'])
        self.set_state(state)
    
class KnowledgePointRAG(AbstractRAG):
    def __init__(self, 
                 model, 
                 sentence_embedder: Callable[[str], np.ndarray],
                 memory_bank: KnowledgePointMemory,
                 min_similarity_score: float):
        super().__init__(model, sentence_embedder, memory_bank)
        self._model = model
        self._embedder = sentence_embedder
        self._memory_bank = memory_bank
        self._stored_hashes = set()
        self._min_similarity_score = min_similarity_score

    # retrieve by similarity
    def retrieve_question_by_similarity(self, question, num_of_question_to_retrieve):
        return self._memory_bank.retrieve_associative_with_text(query = question,
                                                        k = num_of_question_to_retrieve)
    
    # retrieve by threshold
    def retrieve_question_by_threshold(self, question, threshold):
        return self._memory_bank.retrieve_by_similarity_with_threshold_by_text(query = question,
                                                        threshold = threshold)
    
    # retrieve by knowledge points
    def retrieve_question_by_keywords(self, keywords, num_of_question_to_retrieve) -> dict[str:Sequence[str]]:
        return self._memory_bank.retrieve_associative_with_keywords(keywords = keywords,
                                                        k = num_of_question_to_retrieve)
    # add memory
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
    print(rag.retrieve_question_by_similarity('How to reverse a string in Python?', 4))