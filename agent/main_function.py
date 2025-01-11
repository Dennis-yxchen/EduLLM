
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
'nohup ollama serve > ./output.log 2>&1 &'
from concordia.document import interactive_document
from rag_related.extract_knowledge_point import KnowledgePointExtractor
from rag_related.memory_without_time import NaiveAssociativeMemory
from bloom_classifier.bloom_classifier import BloomLevelClassifier
from concordia.language_model import language_model
from rag_related.abstract_rag import AbstractRAG
from rag_related.NaiveRAG import NaiveRAG
from data_utils.dataset_reader import DatasetReader
from config import RAGConfig

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
memory_bank = NaiveAssociativeMemory(
    embedder,
    importance_threshold=RAGConfig.IMPORTANCE_THRESHOLD,
    max_memories=RAGConfig.MAX_MEMORIES,
    deduplication_threshold=RAGConfig.DEDUPLICATION_THRESHOLD,
    contextualize_size=RAGConfig.CONTEXTUALIZE_SIZE,)

rag_tool = NaiveRAG(
    model,
    embedder, 
    memory_bank, 
    min_similarity_score = RAGConfig.MIN_SIMILARITY_SCORE)
bloom_classifier = BloomLevelClassifier(model=model, path = r'./bloom_classifier/definition_of_bloom.json')
knowledge_point_extractor = KnowledgePointExtractor(model, 3)


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
            # bloom_level, bloom_prompt_string = self._bloom_classifier.classify_bloom_level(question)
            # knowledge_point, extract_knowledge_string = self._knowledge_point_extractor.extract_knowledge_point(question)
            # 先用naive rag测试
            self._rag_tool.add_question_to_memory(question)
    
    def _json_to_text(self, question_json):
        pass
    
    def _generate_question_from_pastpaper(self, past_paper):
        new_questions = dict()
        for index, question in enumerate(past_paper):
            print(f"Generating question {index + 1}/{len(past_paper)}")
            prompt = interactive_document.InteractiveDocument(self._model)
            questions = self._rag_tool.retrieve_question_by_similarity(question = question, 
                                                                       num_of_question_to_retrieve = NUM_OF_QUESTION_TO_RETRIEVE)
            examples = "\n".join(questions)
            generating_questions = (
                f"Given the target question '{question}' and the example question '{examples}', generate a new question that is analogous in terms of subject matter and complexity. "
                "Please provide only the new question in your response."
            )
            new_question = prompt.open_question(generating_questions, terminators = ())
            new_questions[index] = new_question
            print(f"Generated question: {new_question}")
            print(f"prompt: {prompt.view().text()}")
            print(f"\n\n\n")
        return new_questions
            
            
            
    
        
        
            
            
            
        
        




if __name__ == "__main__":
    # prompt = interactive_document.InteractiveDocument(model)
    # print("You can eat:", ['KFC', "hotpot", "ice cream"][prompt.multiple_choice_question("what should i eat this afternoon?", answers=['KFC', "hotpot", "ice cream"])])
    # # print(prompt.open_question("what should i eat this afternoon?", terminators = (), answer_prefix="You can eat: "))
    # print(f"embedder: {embedder('what should i eat this afternoon?')}")
    # extractor = KnowledgePointExtractor(model, 3)
    # question = (
    #     "Question: 'We conducted a coin flip consisting of 100 flips, resulting in 61 heads and 39 tails. Our null hypothesis states that 'The coin is fair', meaning that the probability of getting a head is 0.5, and the probability of getting a tail is also 0.5. Your task is to determine whether to accept or reject the null hypothesis. Please support your decision using the p-value of the outcome. You may consult the 'snd' table to obtain an upper bound on the p-value."
    # )
    # print(extractor.extract_knowledge_point(question))
    # memory_bank.add("how are you")
    # memory_bank.add("how are you")
    # memory_bank.add("how are you_1")
    # memory_bank.add("good morning")
    # memory_bank.add("good night")
    # memory_bank.add(question)

    # print("")
    # print(memory_bank.get_all_memories_as_text())
    # print("")
    # print(memory_bank.retrieve_associative("how are you"))
    
    def format_question(questions):
        preprocessed_questions = []
        counter = 0
        for section, type_of_question in questions.items():
            # print(section)
            # print(type_of_question)
            for key,question in type_of_question.items():
                question_string = ""
                for question_component in question:
                    for key, value in question_component.items():
                        if not isinstance(value, list) and not isinstance(value, dict):
                            question_string += f"{key}: {value}\n"
                            # print(f"{counter}: {key}: {value}\n\n")
                        else:
                            if isinstance(value, list):
                                for i in value:
                                    if isinstance(i, str):
                                        question_string += i + "\n"
                                        # print(f"{counter}: {i}\n\n")
                                    else:
                                        for key, value in i.items():
                                            question_string += f"{key}: {value}\n"
                                            # print(f"{counter}: {key}: {value}\n\n")
                                    
                    counter += 1
                    preprocessed_questions.append(question_string)
        return preprocessed_questions
        # print(f"{preprocessed_questions}")
    # reader_1 = DatasetReader('20_fina_1310.json', preprocess_func = format_question)
    # data_1 = reader_1.get_data()
    
    # reader_2 = DatasetReader('21_fina_1310.json', preprocess_func = format_question)
    # data_2 = reader_2.get_data()

    file_names = ['20_fina_1310.json', '21_fina_1310.json', '22_fina_1310.json']
    readers = [DatasetReader(file_name, preprocess_func=format_question) for file_name in file_names]
    data = []
    for reader in readers:
        data.extend(reader.get_data())
    agent = EduLLM_Agent(model, embedder, memory_bank, rag_tool, bloom_classifier, knowledge_point_extractor)
    
    agent._preprocess_past_paper(data)
    
    test_reader = DatasetReader('23_fina_1310.json', preprocess_func = format_question)
    test_data = test_reader.get_data()
    
    result = agent._generate_question_from_pastpaper(test_data)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    with open(f'./output/{timestamp}.json', 'w') as f:
        json.dump(result, f, indent=4)
    
    
    
    
    