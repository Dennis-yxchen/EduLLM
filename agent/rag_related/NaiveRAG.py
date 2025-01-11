from collections.abc import Callable
from typing import Iterable

import numpy as np
from rag_related.memory_without_time import NaiveAssociativeMemory

from rag_related.extract_knowledge_point import KnowledgePointExtractor
from rag_related.abstract_rag import AbstractRAG

class NaiveRAG(AbstractRAG):
    def __init__(self, 
                 model, 
                 sentence_embedder: Callable[[str], np.ndarray],
                 memory_bank: NaiveAssociativeMemory,
                 k: int,
                 similarity_threshold: float,
                 use_importance_weighting: bool,
                 min_similarity_score: float):
        super().__init__(model, sentence_embedder, memory_bank)
        self._model = model
        self._embedder = sentence_embedder
        self._memory_bank = memory_bank
        self._stored_hashes = set()
        self._num_of_question_to_retrieve = k
        self._similarity_threshold = similarity_threshold
        self._use_importance = use_importance_weighting
        self._min_similarity_score = min_similarity_score
            
    def retrieve_question(self, question):
        return self._memory_bank.retrieve_associative(question,
                                                        self._num_of_question_to_retrieve)
    
    def add_question_to_memory(self, question:str, tags:Iterable[str] = ()):
        self._memory_bank.add(text = question, tags = tags)
        
        
    