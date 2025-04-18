from collections import abc
from collections.abc import Callable, Iterable, Sequence
import datetime
import json
import threading

import numpy as np
import pandas as pd
from rag_related.importance_model import ConstantImportanceModel

# _NUM_TO_RETRIEVE_TO_CONTEXTUALIZE_IMPORTANCE = 25
class NaiveAssociativeMemory:
    """Class implementing associative memory"""

    def __init__(self,
                 sentence_embedder : Callable[[str], np.ndarray],
                 importance_threshold: float,
                 max_memories: int,
                 contextualize_size: int,
                 deduplication_threshold: float,
                 importance: Callable[[str, Sequence[tuple[str, float]]], float] | None = None,
                 seed: int | None = None,
                 ):
        """Constructor

        Args:
          sentence_embedder: Text embedding model
          importance: A model that maps sentences to importance in the range [0, 1]. 
                  If None, a constant importance model is used that sets all memories to an importance of 1.0.
          seed: Optional seed used by the random number generator. 
            If None, the default RNG is used.
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

        self._memory_bank = pd.DataFrame(
            columns=['text', 'tags', 'embedding', 'importance']
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
        *,
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

        text = text.replace('\n', ' ')

        contents = {
            'text': text,
            'tags': tuple(tags),
            'importance': importance,
        }
        hashed_contents = hash(tuple(contents.values()))
        derived = {'embedding': self._embedder(text)}
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
        """Add texts to memory

        Args:
          texts: List of strings to add to memory
          **kwargs: Parameters to pass to .add
        """
        for text in texts:
            self.add(text, **kwargs)

    def get_data_frame(self) -> pd.DataFrame:
        with self._memory_bank_lock:
            return self._memory_bank.copy()

    def _get_top_k_cosine(self, x: np.ndarray, k: int):
        """Return the top-k rows most similar to the input vector x.

        Args:
          x: Input vector.
          k: Number of rows to return.

        Returns:
          Rows sorted in descending order of cosine similarity.
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
        """Return the top-k rows most similar to the input vector x.

        Args:
          x: Input vector.
          k: Number of rows to return.
          use_importance: If true, weight similarity by importance.

        Returns:
          Rows sorted in descending order of similarity.
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
        Returns rows with similarity to the input vector x greater than the threshold.
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


    def _pd_to_text(
        self,
        data: pd.DataFrame,
    ) -> Sequence[str]:
        """Format the DataFrame into a list of strings.

        Args:
          data: The DataFrame to process.

        Returns:
          A list of strings, where each memory corresponds to one string.
        """
        output = data['text']
        return output.tolist()

    def retrieve_associative(
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
    
    def retrieve_by_similarity_with_threshold(
        self,
        query: str,
        threshold: float = 0.7,
    ) -> Sequence[str]:
        """Retrieve memories by similarity with a threshold.

        Args:
          query: The query string for retrieval.
          threshold: The similarity threshold for retrieval.

        Returns:
          A list of memory strings that meet the similarity threshold.
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
          A list of memory strings that match the regular expression.
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
          A list of memory strings.
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
          A list of memory strings along with their importance values.
        """
        with self._memory_bank_lock:
            data = self._memory_bank.sample(k, random_state=self._seed)
        return tuple(zip(list(data['text']), list(data['importance'])))

    def __len__(self):
        """Return the number of entries in the memory bank.

        Since memories cannot be deleted, the length will not decrease, 
        and this can be used to check if the content of the memory bank has changed.
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
        """Return the minimum importance of memories in the memory bank."""
        with self._memory_bank_lock:
            return self._memory_bank['importance'].min()

    def set_num_to_retrieve_to_contextualize_importance(
        self, num_to_retrieve: int) -> None:
        """Set the number of memories to retrieve for contextualizing importance.

        Set to 0 to disable contextualization of importance.

        Args:
          num_to_retrieve: Number of memories to retrieve for contextualizing importance.
        """
        self._num_to_retrieve_to_contextualize_importance = num_to_retrieve

    def get_all_memories_as_text(self) -> Sequence[str]:
        """Return all memories in the memory bank as a sequence of strings"""
        memories_data_frame = self.get_data_frame()
        texts = self._pd_to_text(memories_data_frame)
        return texts
    
    def save(self, file_path: str) -> None:
        """Save the memory bank to a JSON file"""
        state = self.get_state()
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)

    def load(self, file_path: str) -> None:
        """Load the memory bank from a JSON file"""
        with open(file_path, 'r') as f:
            state = json.load(f)
        # Convert embedding from list back to np.ndarray
        for item in state['memory_bank']:
            item['embedding'] = np.array(item['embedding'])
        self.set_state(state)