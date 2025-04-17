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
        self._name_to_desc = self._construct_name_description_dict(self._definition)
        
    def classify_bloom_level(self, question):
        prompt = interactive_document.InteractiveDocument(self._model)
        input_prompt = (
            f"{self._bloom_level_string}\n"
            f"Please classify the Bloom's level of the following question:\n"
            f"{question}\n"
            f"Directly answer with the level name: 'Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', or 'Create'.\n"
            f"Please do not add any other information.\n"
        )
        answer = prompt.open_question(input_prompt, terminators=(),max_tokens=2048)
        return answer, prompt.view().text()
    
    def _construct_name_description_dict(self, bloom):
        name_to_desc = {level["name"]: level["description"] for level in bloom["levels"]}
        return name_to_desc
    
    def get_definition(self) -> dict:
        return self._definition
    
    def get_name_description_dict(self) -> dict:
        return self._name_to_desc