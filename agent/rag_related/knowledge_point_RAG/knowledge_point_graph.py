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
from extract_knowledge_point import knowledge_point_extractor
        
class knowledge_memory():
    def __init__(self, model, sentence_embedder, memory_bank, max_knowledge):
        self._model = model
        self._embedder = sentence_embedder
        self._memory_bank = memory_bank
        self._stored_hashes = set()
        self._knowledge_point = []
        self._extractor = knowledge_point_extractor(model, 3).extract_knowledge_point
                
    def add_knowledge_point(self, question):
        knowledge_points, prompt = self._extractor(question)
        self._knowledge_point.extend(knowledge_points.split('\n'))
        self._memory_bank.add(knowledge_points)
        
    def retrieve_associative(self, question):
        pass
        
    