
from typing import Tuple
import matplotlib.pyplot as plt
import sys
from IPython import display
import sentence_transformers
from concordia.language_model import utils
import json
import os
# sys.path.insert(0, os.path.abspath('../../..'))
sys.path.insert(0, os.path.abspath('.'))

from concordia.document import interactive_document
from rag_related.extract_knowledge_point import KnowledgePointExtractor
from rag_related.memory_without_time import NaiveAssociativeMemory
from bloom_classifier.bloom_classifier import BloomLevelClassifier
from concordia.language_model import language_model
from rag_related.abstract_rag import AbstractRAG
from rag_related.NaiveRAG import NaiveRAG
from data_utils.dataset_reader import DatasetReader

st_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2')
embedder = lambda x: st_model.encode(x, show_progress_bar=False)

api_type = 'ollama'
model_name = 'qwen2.5:14b'
disable_language_model = True
model = utils.language_model_setup(
    api_type=api_type,
    model_name=model_name,
    disable_language_model=disable_language_model,
)
memory_bank = NaiveAssociativeMemory(embedder)
rag_tool = NaiveRAG(model, embedder, memory_bank, 3)

class EduLLM_Agent():
    def __init__(self, 
                 model:language_model,
                 embedder: callable,
                 memory_bank: NaiveAssociativeMemory,
                 rag_tool: AbstractRAG,
                 bloom_classifier: BloomLevelClassifier,
                 knowledge_point_extractor: KnowledgePointExtractor):
        self._model = model
        self._embedder = embedder
        self._memory_bank = memory_bank
        self._rag_tool = rag_tool
        self._bloom_classifier = bloom_classifier
        self._knowledge_point_extractor = knowledge_point_extractor
        
    
    
    
    def _preprocess_past_paper(self, past_paper:Tuple[str]):
        """
        each question: 
        1. bloom level classification
        2. extract knowledge point  -> {question, bloom, knowledge}
        3. knowledge point as query, bloom level + few-shot example
        """
        for question in past_paper:
            # 先不用这俩
            bloom_level, bloom_prompt_string = self._bloom_classifier.classify_bloom_level(question)
            knowledge_point, extract_knowledge_string = self._knowledge_point_extractor.extract_knowledge_point(question)
            # 先用naive rag测试
            self._rag_tool.add_question_to_memory(question)
    
    def _json_to_text(self, question_json):
        pass
    
    def _retrieve_question_from_pastpaper(self, past_paper):
        for question in past_paper:
            pass
    
        
        
            
            
            
        
        




if __name__ == "__main__":
    prompt = interactive_document.InteractiveDocument(model)
    print("You can eat:", ['KFC', "hotpot", "ice cream"][prompt.multiple_choice_question("what should i eat this afternoon?", answers=['KFC', "hotpot", "ice cream"])])
    # print(prompt.open_question("what should i eat this afternoon?", terminators = (), answer_prefix="You can eat: "))
    print(f"embedder: {embedder('what should i eat this afternoon?')}")
    extractor = KnowledgePointExtractor(model, 3)
    question = (
        "Question: 'We conducted a coin flip consisting of 100 flips, resulting in 61 heads and 39 tails. Our null hypothesis states that 'The coin is fair', meaning that the probability of getting a head is 0.5, and the probability of getting a tail is also 0.5. Your task is to determine whether to accept or reject the null hypothesis. Please support your decision using the p-value of the outcome. You may consult the 'snd' table to obtain an upper bound on the p-value."
    )
    print(extractor.extract_knowledge_point(question))
    memory_bank.add("how are you")
    memory_bank.add("how are you")
    memory_bank.add("how are you_1")
    memory_bank.add("good morning")
    memory_bank.add("good night")
    memory_bank.add(question)

    print("")
    print(memory_bank.get_all_memories_as_text())
    print("")
    print(memory_bank.retrieve_associative("how are you"))
    
    
    