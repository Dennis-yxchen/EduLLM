from abc import ABC, abstractmethod

class AbstractRAG(ABC):
    def __init__(self, model, embedder, memory_bank):
        super().__init__()
        self._model: None = None
        self._embedder: None = None
        self._memory_bank: None = None
    @abstractmethod
    def retrieve_question(self, question:str):
        raise NotImplementedError
    
    @abstractmethod
    def add_question_to_memory(self, question:str,tags:tuple | None=None):
        raise NotImplementedError