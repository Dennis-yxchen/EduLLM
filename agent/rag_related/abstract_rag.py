from abc import ABC, abstractmethod
from typing import Iterable

class AbstractRAG(ABC):
    def __init__(self, model, embedder, memory_bank):
        super().__init__()
        self._model: None = None
        self._embedder: None = None
        self._memory_bank: None = None
    
    @abstractmethod
    def retrieve_question_by_similarity(self, question:str, num_of_question_to_retrieve:int):
        raise NotImplementedError
    
    @abstractmethod
    def retrieve_question_by_threshold(self, question:str, threshold:float):
        raise NotImplementedError
    
    @abstractmethod
    def add_question_to_memory(self, question:str,tags:Iterable[str] = ()):
        raise NotImplementedError