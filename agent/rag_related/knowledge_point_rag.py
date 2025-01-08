# class knowledge_graph_builder():
#     def __init__(self, model, sentence_embedder):
#         self._model = model
#         self._embedder = sentence_embedder
#         self._stored_hashes = set()
#         self.node_list = []
        
#     def build_knowledge_graph(self, whole_question_list):
#         prompt = interactive_document.InteractiveDocument(self._model)
#         hash_table = {}
#         for i in whole_question_list:
#             hash_table[i] = self.
import sys
import os

# 添加EduLLM目录到sys.path
from memory_without_time import NaiveAssociativeMemory

from extract_knowledge_point import KnowledgePointExtractor
from abstract_rag import AbstractRAG

class knowledge_memory(AbstractRAG):
    def __init__(self, model, sentence_embedder, memory_bank: NaiveAssociativeMemory, max_knowledge: int, threshold: float):
        super().__init__(model, sentence_embedder, memory_bank)
        self._model = model
        self._embedder = sentence_embedder
        self._memory_bank = memory_bank
        self._stored_hashes = set()
        self._knowledge_point = []
        self._extractor = KnowledgePointExtractor(model, max_knowledge).extract_knowledge_point
        self._threshold = threshold
                
    def add_knowledge_point_from_question(self, question, tags:tuple | None=None):
        knowledge_points, prompt = self._extractor(question)
        self._knowledge_point.extend(knowledge_points.split('\n'))
        # 可以尝试用其他方法 e.g. 直接将knowledge point embed 进去，然后把所有knowledge point的embedding算cosine similarity
        self._memory_bank.add(knowledge_points, tags=tags) # buggy: 有可能不同question有完全相同的knowledge point，但是这里会跳过
        # 可能得改memory, 让它能接受多个embedding，然后在retrieve的时候, 选择特定的embedding进行计算
        
    
        
        
    def retrieve_associative_with_threshold(self, question):
        knowledge_points, prompt = self._extractor(question)
        related_questions = self._memory_bank.retrieve_by_similarity_with_threshold(knowledge_points, self._threshold)
        return related_questions
    
    def retrieve_question(self, question):
        return self.retrieve_associative_with_threshold(question)
    
    def add_question_to_memory(self, question, tags:tuple | None=None):
        self.add_knowledge_point_from_question(question)
    
        
    