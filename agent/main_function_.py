
from typing import Tuple
import matplotlib.pyplot as plt
import sys
from IPython import display
import sentence_transformers
from concordia.language_model import utils
import json
import os
from concordia.document import interactive_document
from rag_related.extract_knowledge_point import KnowledgePointExtractor
from rag_related.memory_without_time import NaiveAssociativeMemory
from bloom_classifier.bloom_classifier import BloomLevelClassifier
from concordia.language_model import language_model
from rag_related.abstract_rag import AbstractRAG
from rag_related.NaiveRAG import NaiveRAG
from rag_related.KnowledgePointRAG import KnowledgePointRAG, KnowledgePointMemory
from data_utils.dataset_reader import DatasetReader
from config import RAGConfig
from tqdm import tqdm
from data_utils.data_output import generate_pdf_from_json
from models import get_model
from concordia.language_model import no_language_model

st_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2')
embedder = lambda x: st_model.encode(x, show_progress_bar=False)

print(f"get the embedder")

disable_language_model = False
if disable_language_model:
    model = no_language_model.NoLanguageModel()
else:
    model = get_model(
        model_name="deepseek-ai/DeepSeek-V3",
        # model_name='Qwen/Qwen2.5-14B-Instruct',
        api_key="sk-ufvfjzrydqzznjfnqabneayuhyimirhnwekmiemjyskvxedo",
    )

print(f"get the model")

# the memory bank
memory_bank = KnowledgePointMemory(
        sentence_embedder=embedder,
        importance_threshold=0.5,
        max_memories=1000,
        contextualize_size=5,
        deduplication_threshold=0.1,
        num_knowledge_points=3
    )

# the wrapper for the rag function
rag_tool = KnowledgePointRAG(
        model=model,  # Replace with your actual model
        sentence_embedder=embedder,
        memory_bank=memory_bank,
        min_similarity_score=0.7
    )


DIR_NAME = os.path.dirname(os.path.abspath(__file__))

# the bloom classifier (EduLLM w/ Bloom)
bloom_classifier = BloomLevelClassifier(model=model, path = os.path.join(DIR_NAME, r'bloom_classifier/definition_of_bloom.json'))

# knowledge point extractor for the EduLLM (Knowledge Point retrieval)
knowledge_point_extractor = KnowledgePointExtractor(model, 3)


class EduLLM_Agent():
    def __init__(self, 
                 model:language_model,
                 embedder: callable,
                 memory_bank: NaiveAssociativeMemory,
                 rag_tool: KnowledgePointRAG,
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
            bloom_level, bloom_prompt_string = self._bloom_classifier.classify_bloom_level(question)
            knowledge_point, extract_knowledge_string = self._knowledge_point_extractor.extract_knowledge_point(question)
            print(f"question: {question}")
            print(f"bloom level: {bloom_level}")
            print(f"knowledge point: {knowledge_point}")
            print(f"\n\n\n")
            
            # add question to the memory with knowledge points
            self._rag_tool.add_question_to_memory(question = question, knowledge_points = knowledge_point)
    
    # Naive RAG
    def _generate_question_from_pastpaper_vanilla(self, past_paper):
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
            new_question = prompt.open_question(generating_questions, terminators=(), max_tokens = 4096)
            new_questions[index] = new_question
            print(f"Generated question: {new_question}")
            # print(f"prompt: {prompt.view().text()}")
            print(f"\n\n\n")
        return new_questions

    # direct generation
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
            new_question = prompt.open_question(generating_questions, terminators = (), max_tokens = 4096)
            new_questions[index] = new_question
            print(f"Generated question: {new_question}")
            # print(f"prompt: {prompt.view().text()}")
            print(f"\n\n\n")
        return new_questions
    
    # EduLLM with Knowledge points, but without question, generate one question
    def _generate_question_by_knowledge_point_and_question(self,question):
        prompt = interactive_document.InteractiveDocument(self._model)
        # 1. extract knowledge point
        knowledge_point, knowledge_prompt_string = self._knowledge_point_extractor.extract_knowledge_point(question)
        # 2. retrieve question by knowledge point
        questions_from_keywords_dict = self._rag_tool.retrieve_question_by_keywords(keywords=knowledge_point, 
                                                                    num_of_question_to_retrieve = RAGConfig.NUM_RETRIEVED_DOCS)

        
        
        formatted_data = [
            f"{text}\nknowledge points: {','.join(knowledge_points)}\n"
            for text, knowledge_points in zip(questions_from_keywords_dict['text'], questions_from_keywords_dict['knowledge_points'])
        ]
        
        # 3. generate question
        generating_questions = (
                f"Given knowledge points that this question want to assess:\n"
                f"Knowledge point: \n{','.join(knowledge_point)}\n"
                f"Some example questions with related knowledge points:\n"
                f"{'\n'.join(formatted_data)}\n"
                "generate a new question that is analogous in terms of subject matter and complexity. "
                "Please provide only the new question in your response."
            )
        
        new_question = prompt.open_question(generating_questions, terminators=(),
                                            max_tokens = 4096,)
        return new_question, prompt.view().text()
        
        
    # Generate whole paper with _generate_question_by_knowledge_point_and_question
    def _generate_question_from_pastpaper_with_knowledge_point(self, past_paper):
        # idea: 
        new_questions = dict()
        question_with_index = enumerate(past_paper)
        import concurrent.futures
        max_workers = min(8, len(past_paper))
        
        
        ## iterative version
        # for index, question in tqdm(question_with_index, desc="Generating questions"):
        #     print(f"Generating question {index + 1}/{len(past_paper)}")
        #     new_question, prompt_string = self._generate_question_by_knowledge_point_and_question(question)
        #     new_questions[index] = new_question
        #     print(f"Generated question: {new_question}")
        #     print(f"prompt: {prompt_string}")
        #     print(f"\n\n\n")
        # return new_questions
        
        # concurrent version
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {}
            
            for index, question in question_with_index:
                print(f"Submitting question {index + 1}/{len(past_paper)} for generation")
                # Submit task to thread pool
                future = executor.submit(
                    self._generate_question_by_knowledge_point_and_question, 
                    question
                )
                future_to_index[future] = index
            
            # Initialize the result dictionary
            new_questions = {}
            
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    new_question, prompt_string = future.result()
                    new_questions[index] = new_question
                    print(f"Generated question {index + 1}: {new_question}")
                    print(f"prompt: {prompt_string}")
                    print("\n\n\n")
                except Exception as exc:
                    print(f'Question {index + 1} generated an exception: {exc}')
            
            # Sort results by original index
            sorted_indices = sorted(new_questions.keys())
            sorted_questions = {i: new_questions[i] for i in sorted_indices}
            
            return sorted_questions

    # Generate one question by EduLLM (our method) and with bloom level
    def _generate_question_by_knowledge_point_and_question_and_bloom(self, question):
        prompt = interactive_document.InteractiveDocument(self._model)
        # 1. extract knowledge point
        knowledge_point, knowledge_prompt_string = self._knowledge_point_extractor.extract_knowledge_point(question)
        # 2. retrieve question by knowledge point
        questions_from_keywords_dict = self._rag_tool.retrieve_question_by_keywords(keywords=knowledge_point, 
                                                                    num_of_question_to_retrieve = RAGConfig.NUM_RETRIEVED_DOCS)
        
        print(questions_from_keywords_dict)
        
        # 3. classify bloom level
        bloom_level, bloom_prompt_string = self._bloom_classifier.classify_bloom_level(question)
        print(f"bloom level: {bloom_level}")
        
        formatted_data = [
            f"{text}\nknowledge points: {','.join(knowledge_points)}\n"
            for text, knowledge_points in zip(questions_from_keywords_dict['text'], questions_from_keywords_dict['knowledge_points'])
        ]
        
        bloom_level_dict = self._bloom_classifier.get_name_description_dict()
        bloom_level_description = bloom_level_dict[bloom_level.strip(' \'\",;.*?')]
        
        print(f"bloom level description: {bloom_level_description}")
        
        
        # 3. generate question
        generating_questions = (
                f"Given knowledge points that this question want to assess:\n"
                f"Knowledge point: \n{','.join(knowledge_point)}\n"
                f"Some example questions with related knowledge points:\n"
                f"{'\n'.join(formatted_data)}\n"
                "generate a new question that is analogous in terms of subject matter and complexity. \n"
                f"The bloom level of the new question should be {bloom_level}.\n"
                f"Which means {bloom_level_description}.\n"
                "Please provide only the new question in your response."
            )
        
        new_question = prompt.open_question(generating_questions, terminators=(),
                                            max_tokens = 4096,)
        return new_question, prompt.view().text()

    # Generate whole paper with _generate_question_by_knowledge_point_and_question_and_bloom
    def _generate_question_from_pastpaper_with_knowledge_point_and_bloom(self, past_paper):
        # idea: 
        new_questions = dict()
        question_with_index = enumerate(past_paper)
        import concurrent.futures
        max_workers = min(8, len(past_paper))
        for index, question in tqdm(question_with_index, desc="Generating questions"):
            print(f"Generating question {index + 1}/{len(past_paper)}")
            new_question, prompt_string = self._generate_question_by_knowledge_point_and_question_and_bloom(question)
            new_questions[index] = new_question
            print(f"Generated question: {new_question}")
            print(f"prompt: {prompt_string}")
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

    
    print(f"preprocessing datasets")
    file_names = ['20_fina_1310.json', '21_fina_1310.json']
    readers = [DatasetReader(data_path='..\\dataset', file_name=file_name, preprocess_func=format_question) for file_name in file_names]
    data = []
    for reader in readers:
        data.extend(reader.get_data())
    agent = EduLLM_Agent(model, embedder, memory_bank, rag_tool, bloom_classifier, knowledge_point_extractor)
    agent._preprocess_past_paper(data)
    
    print(f"generating dataset")
    test_reader = DatasetReader(data_path='..\\dataset', file_name='23_fina_1310.json', preprocess_func = format_question)
    test_data = test_reader.get_data()
    
    from datetime import datetime
    import time
    time.sleep(5)

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    
    
    # select the method here
    ###
    # _generate_question_from_pastpaper_with_knowledge_point_and_bloom(pastpaper)
    # _generate_question_from_pastpaper_with_knowledge_point(pastpaper)
    # _generate_question_from_pastpaper_vanilla(pastpaper)
    # _generate_question_directly(pastpaper)
    ###
    
    suffix = 'KP_bloom'
    result_direct = agent._generate_question_from_pastpaper_with_knowledge_point_and_bloom(test_data)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    with open(f'./output/{timestamp}_{suffix}.json', 'w') as f:
        json.dump(result_direct, f, indent=4)
    generate_pdf_from_json(f"./output/{timestamp}_{suffix}.json", f"./output/{timestamp}_{suffix}.pdf", title = "FINA1310", footnotes=f"Generated time: {timestamp}")
