from concordia.document import interactive_document

class KnowledgePointExtractor():
    # assume input is question
    def __init__(self, model, max_knowledge):
        self._model = model
        self._max_knowledge = max_knowledge
        
    
    # extract the knowledge point from question
    def extract_knowledge_point(self, original_question, learning_outcome:tuple[str]=None):
        # this question assess which learning outcome?
        question = (
            f"{original_question}"
            f"You should output at most {self._max_knowledge} knowledge points that assessed in this question."
            f"output in numbered list format:\n"
            f"1. <knowledge point 1>\n"
            f"2. <knowledge point 2>"
        )
        
        if learning_outcome:
            learning_outcome_string = ', '.join(learning_outcome)
            learning_outcome_question = (
                f"Given the learning outcome: \n{learning_outcome_string}.\n"
            )
            question = learning_outcome_question + question
                
        
        prompt = interactive_document.InteractiveDocument(self._model)
        answer = prompt.open_question(question, terminators=(),)
        
        concerntate_question = (
            "Extract exactly 3 distinct knowledge points from the input.\n"
            "Format requirements:\n"
            "1. Output ONLY the knowledge points\n"
            "2. Use bullet points with - prefix (no numbers)\n"
            "3. Ensure each point is a concise keyword/phrase\n"
            "4. Avoid using punctuation in the points\n"
            "5. Maintain original terminology\n"
            "Example format:\n"
            "- Knowledge point 1\n"
            "- Knowledge point 2\n"
            "- Knowledge point 3"
        )
        import re
        concerntate_answer = prompt.open_question(concerntate_question, terminators=(),)
        
        pattern = r'-\s*([^\n]+)'
        matches = re.findall(pattern, concerntate_answer)
        knowledge_points = []
        for match in matches[:3]:  # Enforce 3 points max
            cleaned = match.strip().rstrip(';.,').lstrip('- ')  # Remove trailing punctuation
            if cleaned:
                knowledge_points.append(cleaned)
        return knowledge_points, prompt.view().text()