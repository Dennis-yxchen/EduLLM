
from concordia.document import interactive_document


class knowledge_point_extractor():
    # assume input is question
    def __init__(self, model, max_knowledge):
        self._model = model
        self._max_knowledge = max_knowledge
        
        
    def extract_knowledge_point(self, Question, learning_outcome=None):
        # this question assess which learning outcome?
        question = (
            f"{Question}"
            f"You should output at most {self._max_knowledge} knowledge points that assessed in this question."
            f"output in numbered list format:\n"
            f"1. <knowledge point 1>\n"
            f"2. <knowledge point 2>"
        )
        
        prompt = interactive_document.InteractiveDocument(self._model)
        answer = prompt.open_question(question, terminators=(),)
        
        concerntate_question = (
            'You should only output knowledge points without any other information. '
            'You should return only with a few keywords that represent the knowledge point.'
        )
        
        concerntate_answer = prompt.open_question(concerntate_question, terminators=(),)
        return concerntate_answer, prompt.view().text()