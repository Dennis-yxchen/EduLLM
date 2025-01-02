import collections
import concurrent.futures
import datetime
import random
import matplotlib.pyplot as plt
import sys
from IPython import display
import sentence_transformers
from collections.abc import Callable, Sequence
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.language_model.language_model import LanguageModel
from concordia.utils import measurements as measurements_lib
from concordia.utils import html as html_lib
from concordia.utils import plotting
from concordia.language_model import utils
import json
import os
## setting start here
from concordia.typing.entity_component import EntityWithComponents
from concordia.document import interactive_document

from extract_knowledge_point import knowledge_point_extractor
from EduLLM.agent.memory_without_time import NaiveAssociativeMemory
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




if __name__ == "__main__":
    prompt = interactive_document.InteractiveDocument(model)
    print("You can eat:", ['KFC', "hotpot", "ice cream"][prompt.multiple_choice_question("what should i eat this afternoon?", answers=['KFC', "hotpot", "ice cream"])])
    # print(prompt.open_question("what should i eat this afternoon?", terminators = (), answer_prefix="You can eat: "))
    print(f"embedder: {embedder('what should i eat this afternoon?')}")
    extractor = knowledge_point_extractor(model, 3)
    question = (
        "Question: 'We conducted a coin flip consisting of 100 flips, resulting in 61 heads and 39 tails. Our null hypothesis states that 'The coin is fair', meaning that the probability of getting a head is 0.5, and the probability of getting a tail is also 0.5. Your task is to determine whether to accept or reject the null hypothesis. Please support your decision using the p-value of the outcome. You may consult the 'snd' table to obtain an upper bound on the p-value."
    )
    print(extractor.extract_knowledge_point(question))
    memory_bank.add("how are you")
    memory_bank.add("how are you")
    memory_bank.add("how are you_1")


    print(memory_bank.retrieve_associative("how are you"))
    print(memory_bank.get_all_memories_as_text())
    