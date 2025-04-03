
from typing import Tuple
import matplotlib.pyplot as plt
import sys
from IPython import display
import sentence_transformers
from concordia.language_model import utils
import json
import os


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
from tqdm import tqdm
from data_utils.data_output import generate_pdf_from_json
from models import get_model
from concordia.language_model import no_language_model

st_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2')
embedder = lambda x: st_model.encode(x, show_progress_bar=False)

api_type = 'ollama'
model_name = 'qwen2.5:14b'
disable_language_model = False
# model = utils.language_model_setup(
#     api_type=api_type,
#     model_name=model_name,
#     disable_language_model=disable_language_model,
# )
if disable_language_model:
    model = no_language_model.NoLanguageModel()
else:
    model = get_model(
        # model_name="deepseek-ai/DeepSeek-V3",
        model_name='Qwen/Qwen2.5-14B-Instruct',
        api_key="sk-ufvfjzrydqzznjfnqabneayuhyimirhnwekmiemjyskvxedo",
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
        for index, question in enumerate(tqdm(past_paper, desc="Generating questions")):
            print(f"Generating question {index + 1}/{len(past_paper)}")
            prompt = interactive_document.InteractiveDocument(self._model)
            questions = self._rag_tool.retrieve_question_by_similarity(question = question, 
                                                                       num_of_question_to_retrieve = RAGConfig.NUM_RETRIEVED_DOCS)
            # questions = self._rag_tool.retrieve_question_by_threshold(question = question, 
                                                                    #    threshold = RAGConfig.SIMILARITY_THRESHOLD)
            examples = "\n".join(questions)
            generating_questions = (
                f"Given the target question '{question}' and the example question '{examples}', "
                "generate a new question that is analogous in terms of subject matter and complexity. "
                "Please provide only the new question in your response."
            )
            new_question = prompt.open_question(generating_questions, terminators=())
            new_questions[index] = new_question
            print(f"Generated question: {new_question}")
            # print(f"prompt: {prompt.view().text()}")
            print(f"\n\n\n")
        return new_questions
            
    def _generate_question_directly(self, past_paper):
        new_questions = dict()
        for index, question in enumerate(tqdm(past_paper, desc="Generating questions")):
            print(f"Generating question {index + 1}/{len(past_paper)}")
            prompt = interactive_document.InteractiveDocument(self._model)
            generating_questions = (
                f"Given the target question '{question}', "
                "generate a new question that is analogous in terms of subject matter and complexity. "
                "Please provide only the new question in your response."
            )
            new_question = prompt.open_question(generating_questions, terminators = ())
            new_questions[index] = new_question
            print(f"Generated question: {new_question}")
            # print(f"prompt: {prompt.view().text()}")
            print(f"\n\n\n")
        return new_questions
            
    
        
        
            
            
            
        
        




if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    def format_question(questions):
        combined_questions = []

        # Iterate through each item in the data list
        for item in questions:
            # Extract the components
            question = item['question']
            q_type = item['type']
            options = item.get('options', None)  # Use get to handle missing keys
            
            # Start building the formatted string
            formatted_str = f"Question: {question}\nType: {q_type}\n"
            
            if options is not None:
                # Join the options with newline and indentation
                options_str = '\n    '.join(options)
                formatted_str += f"Options:\n    {options_str}\n"
            else:
                formatted_str += "Options: No options provided.\n"
            
            # Append the formatted string to the list
            combined_questions.append(formatted_str)
        return combined_questions

    file_names = ['20_fina_1310.json', '21_fina_1310.json']
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
    import time

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    with open(f'./output/{timestamp}.json', 'w') as f:
        json.dump(result, f, indent=4)
    generate_pdf_from_json(f"./output/{timestamp}.json", f"./output/{timestamp}.pdf", title = "FINA1310", footnotes=f"Generated time: {timestamp}")
    
    # Sleep for 5 seconds
    time.sleep(5)

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    result_direct = agent._generate_question_directly(test_data)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    with open(f'./output/{timestamp}_direct.json', 'w') as f:
        json.dump(result_direct, f, indent=4)
    generate_pdf_from_json(f"./output/{timestamp}_direct.json", f"./output/{timestamp}_direct.pdf", title = "FINA1310", footnotes=f"Generated time: {timestamp}")

    
    
    
    
    
