import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
print(sys.path)

from main_function_ import get_model
from data_utils.dataset_reader import DatasetReader
import tqdm
import openai

model = get_model(
            model_name="deepseek-ai/DeepSeek-V3",
            # model_name='Qwen/Qwen2.5-14B-Instruct',
            api_key="sk-ufvfjzrydqzznjfnqabneayuhyimirhnwekmiemjyskvxedo",
        )

# evaluate the question by LLM as a judge
def evaluate_questions(question_pairs, goal_prompt, role_prompt):
    # Initialize counters
    baseline_wins = {"Completeness": 0, "Knowledge Alignment": 0, "Diversity": 0, "Overall": 0}
    EDULLM_wins = {"Completeness": 0, "Knowledge Alignment": 0, "Diversity": 0, "Overall": 0}
    ties = {"Completeness": 0, "Knowledge Alignment": 0, "Diversity": 0, "Overall": 0}
    
    results = []
    
    for i, pair in tqdm.tqdm(enumerate(question_pairs, 1), desc="Evaluating questions", total=len(question_pairs)):
        question, generated_question_baseline, generated_question_EDULLM = pair
        
        # Prepare the prompt
        current_prompt = goal_prompt.replace('<<original question>>', question)
        current_prompt = current_prompt.replace('<<answer1>>', generated_question_baseline)
        current_prompt = current_prompt.replace('<<answer2>>', generated_question_EDULLM)
        # print(current_prompt)
        # Get model response
        model_response = model.sample_text(
            prompt=current_prompt,
            temperature=0.0,
            max_tokens=8192,
            terminators=(),
            system_prompt=role_prompt,
        )
        
        if model_response.startswith('`') or model_response.endswith('`'):
            model_response = model_response.strip('`')
            if model_response[:4].upper() == 'JSON':
                model_response = model_response[4:]

        
        try:
            # Parse the JSON response
            evaluation = json.loads(model_response)
            
            # Count wins for each criterion
            for criterion in ["Completeness", "Knowledge Alignment", "Diversity"]:
                winner = evaluation[criterion]["Winner"]
                if winner == "Answer 1":
                    baseline_wins[criterion] += 1
                elif winner == "Answer 2":
                    EDULLM_wins[criterion] += 1
                else:
                    ties[criterion] += 1
            
            # Determine overall winner (simple majority)
            baseline_score = sum(1 for criterion in ["Completeness", "Knowledge Alignment", "Diversity"] 
                            if evaluation[criterion]["Winner"] == "Answer 1")
            EDULLM_score = sum(1 for criterion in ["Completeness", "Knowledge Alignment", "Diversity"] 
                            if evaluation[criterion]["Winner"] == "Answer 2")
            
            if baseline_score > EDULLM_score:
                overall_winner = "Answer 1"
                baseline_wins["Overall"] += 1
            elif EDULLM_score > baseline_score:
                overall_winner = "Answer 2"
                EDULLM_wins["Overall"] += 1
            else:
                overall_winner = "Tie"
                ties["Overall"] += 1
            
            # Add overall winner to the evaluation
            evaluation["Overall"] = {"Winner": overall_winner}
            
            results.append({
                "Question": question,
                "Baseline": generated_question_baseline,
                "EDULLM": generated_question_EDULLM,
                "Evaluation": evaluation
            })
            
            # store the backup evaluation
            with open(f'./result_0417/evaluation.json', 'a', encoding='utf-8') as f:
                json.dump({
                    'index': i,
                    "Question": question,
                    "Baseline": generated_question_baseline,
                    "EDULLM": generated_question_EDULLM,
                    "Evaluation": evaluation
                }, f, ensure_ascii=False)
                f.write('\n')
            
        except openai.APIError as e:
            # Handle OpenAI timeout error
            results.append({
                'index': i,
                "Question": question,
                "Baseline": generated_question_baseline,
                "EDULLM": generated_question_EDULLM,
                "Error": "OpenAI Timeout Error",
                "RawResponse": str(e)
            })
            with open(f'./result_0417/evaluation_error.json', 'a', encoding='utf-8') as f:
                json.dump({
                    'index': i,
                    "Question": question,
                    "Baseline": generated_question_baseline,
                    "EDULLM": generated_question_EDULLM,
                    "Error": "OpenAI Timeout Error",
                    "RawResponse": str(e)
                }, f, ensure_ascii=False)
                f.write('\n')

        except json.JSONDecodeError as e:
            # Handle JSON parsing error
            results.append({
                'index': i,
                "Question": question,
                "Baseline": generated_question_baseline,
                "EDULLM": generated_question_EDULLM,
                "Error": "Failed to parse evaluation",
                "RawResponse": model_response
            })
            with open(f'./result_0417/evaluation_error.json', 'a', encoding='utf-8') as f:
                json.dump({
                    'index': i,
                    "Question": question,
                    "Baseline": generated_question_baseline,
                    "EDULLM": generated_question_EDULLM,
                    "Error": "Failed to parse evaluation",
                    "RawResponse": model_response
                }, f, ensure_ascii=False)
                f.write('\n')
    
    # Print summary statistics
    print("\n=== Final Statistics ===")
    print(f"Total question pairs evaluated: {len(question_pairs)}")
    print("\nBaseline wins:")
    for criterion, count in baseline_wins.items():
        print(f"{criterion}: {count}")
    
    print("\nEDULLM wins:")
    for criterion, count in EDULLM_wins.items():
        print(f"{criterion}: {count}")
    
    print("\nTies:")
    for criterion, count in ties.items():
        print(f"{criterion}: {count}")
    
    return results

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
    
def get_original_question(path):
    dataset_reader = DatasetReader(data_path=path, file_name='23_fina_1310.json', preprocess_func = format_question)
    test_data = dataset_reader.get_data()
    return test_data

def get_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    
    data_list = []
    for key in data_dict.keys():
        data_list.append((data_dict[key]))
    
    return data_list


if __name__ == "__main__":
    role_prompt = ''
    ABS_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(ABS_DIR, 'role_prompt.txt'), 'r', encoding='utf-8') as f:
        role_prompt = f.read()
        
    goal_prompt = ''
    with open(os.path.join(ABS_DIR, 'goal_prompt.txt'), 'r', encoding='utf-8') as f:
        goal_prompt = f.read()
        
    original_path = r'./result_0417'
    original_question = get_original_question(original_path)        
    
    baseline = get_data(os.path.join(original_path, 'vanillaRAG.json'))
    EduLLM_KP = get_data(os.path.join(original_path, 'KP.json'))
    
    print("Original question length: ", len(original_question))
    print("Baseline length: ", len(baseline))
    print("EduLLM_KP length: ", len(EduLLM_KP))
    result = evaluate_questions(tuple(zip(
        original_question, EduLLM_KP, baseline
        )), goal_prompt, role_prompt)
    
    import pandas as pd
    df = pd.DataFrame(result)
    df.to_csv(fr'{ABS_DIR}\evaluation_results_VRAG_VS_KPB.csv', index=False)
    print("Evaluation results saved to evaluation_results.csv")