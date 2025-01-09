from concordia.document import interactive_document
import json
def get_bloom_level_definition(path):
    with open(path, 'r') as f:
        definition = json.load(f)
    return definition

def reformat_bloom_level_definition(definition):
    result = []
    for item in definition["levels"]:
        # 使用 f-string 格式化字符串
        formatted_string = f"{item['name']}: This level involves {item['description']}"
        result.append(formatted_string)
    
    result = "\n".join(result)
    result = (
        "Bloom’s Taxonomy offers a framework for categorizing the depth of learning, and it provides guidance on selecting appropriate action verbs when writing learning objectives. Here are the six levels of Bloom’s taxonomy and their definitions:\n"
        f"{result}"
    )
    return result
    

class BloomLevelClassifier():
    def __init__(self, model, path):
        self._model = model
        self._definition = get_bloom_level_definition(path)
        self._bloom_level_string = reformat_bloom_level_definition(self._definition)
        
    def classify_bloom_level(self, question):
        prompt = interactive_document.InteractiveDocument(self._model)
        input_prompt = (
            f"{self._bloom_level_string}\n"
            f"Please classify the Bloom's level of the following question:\n"
            f"{question}"
        )
        answer = prompt.open_question(question, terminators=(),)
        return answer, prompt.view().text()