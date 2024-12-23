# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for language model sampling.
"""

import re


def _extract_parenthesized_choice(sample: str):
  """Given text formatted as 'lorum(a)ipsum', return 'a'."""
  match = re.search(r'\(?(\w)\)', sample)
  if match:
    return match.group(1)
  else:
    return None


def extract_choice_response(sample: str) -> str | None:
  """Given a sample such as "a", "a)", or "foo(a)bar, return the choice."""
  if len(sample) == 1:
    # i.e. this would be a sample such as "a"
    return sample
  elif len(sample) == 2:
    # i.e. this would be a sample such as "a)"
    return sample[0]
  else:
    # extract a substring like "(a)" wherever it may be in a longer string
    return _extract_parenthesized_choice(sample)


def dynamically_adjust_temperature(
    attempts: int,
    max_attempts: int,
) -> float:
  """Adjusts choice sampling temperature based on number of attempts so far."""
  # Increase temperature after the first failed attempt.
  temperature = 0.0
  if attempts > 1 and attempts < (max_attempts / 2.0):
    temperature = 0.5
  elif attempts > (max_attempts / 2.0):
    temperature = 0.75
  return temperature

class InvalidResponseError(Exception):
  """Exception to throw when exceeding max attempts to get a choice."""
  pass


# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DEFAULT_TEMPERATURE = 0.5
DEFAULT_TERMINATORS = ()
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_TOKENS = 256

"""Ollama Language Model, a wrapper for models running on the local machine."""

from collections.abc import Collection, Sequence
import json

import ollama
from typing_extensions import override


_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20
_DEFAULT_TEMPERATURE = 0.5
_DEFAULT_TERMINATORS = ()
_DEFAULT_SYSTEM_MESSAGE = (
    'Continue the user\'s sentences. Never repeat their starts. For example, '
    'when you see \'Bob is\', you should continue the sentence after '
    'the word \'is\'. Here are some more examples: \'Question: Is Jake a '
    'turtle?\nAnswer: Jake is \' should be completed as \'not a turtle.\' and '
    '\'Question: What is Priya doing right now?\nAnswer: Priya is currently \' '
    'should be completed as \'working on repairing the sink.\'. Notice that '
    'it is OK to be creative with how you finish the user\'s sentences. The '
    'most important thing is to always continue in the same style as the user.'
)


class OllamaLanguageModel:
  """Language Model that uses Ollama LLM models."""

  def __init__(
      self,
      model_name: str,
      *,
      system_message: str = _DEFAULT_SYSTEM_MESSAGE,
  ) -> None:
    """Initializes the instance.

    Args:
        model_name: The language model to use. For more details, see
          https://github.com/ollama/ollama.
        system_message: System message to prefix to requests when prompting the
          model.
        measurements: The measurements object to log usage statistics to.
        channel: The channel to write the statistics to.
    """
    self._model_name = model_name
    self._client = ollama.Client()
    self._system_message = system_message
    self._terminators = []

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = _DEFAULT_TERMINATORS,
      temperature: float = _DEFAULT_TEMPERATURE,
      timeout: float = -1,
      seed: int | None = None,
  ) -> str:
    del max_tokens, timeout, seed, temperature  # Unused.

    prompt_with_system_message = f'{self._system_message}\n\n{prompt}'

    terminators = self._terminators + list(terminators)

    response = self._client.generate(
        model=self._model_name,
        prompt=prompt_with_system_message,
        options={'stop': terminators},
        keep_alive='10m',
    )
    result = response['response']

    return result

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    del seed  # Unused.
    prompt_with_system_message = f'{self._system_message}\n\n{prompt}'
    template = {'choice': '', 'single sentence explanation': ''}
    sample = ''
    answer = ''
    for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      # Increase temperature after the first failed attempt.
      temperature = dynamically_adjust_temperature(
          attempts, _MAX_MULTIPLE_CHOICE_ATTEMPTS)

      response = self._client.generate(
          model=self._model_name,
          prompt=(f'{prompt_with_system_message}.\n'
                  f'Use the following json template: {json.dumps(template)}.'),
          options={'stop': (), 'temperature': temperature},
          format='json',
          keep_alive='10m',
      )
      try:
        json_data_response = json.loads(response['response'])
      except json.JSONDecodeError:
        continue
      sample_or_none = json_data_response.get('choice', None)
      if sample_or_none is None:
        if isinstance(json_data_response, dict) and json_data_response:
          sample = next(iter(json_data_response.values()))
        elif isinstance(json_data_response, str) and json_data_response:
          sample = sample_or_none.strip()
        else:
          continue
      else:
        sample = sample_or_none
        if isinstance(sample, str) and sample:
          sample = sample.strip()

      answer = extract_choice_response(sample)
      try:
        idx = responses.index(answer)
      except ValueError:
        continue
      else:
        debug = {}
        return idx, responses[idx], debug

    raise InvalidResponseError(
        (f'Too many multiple choice attempts.\nLast attempt: {sample}, ' +
         f'extracted: {answer}')
    )
    
    
if __name__ == "__main__":
  cls = OllamaLanguageModel("qwen2.5:14b")

  my_input = None
  while True:
      my_input = input("input your input (type 'end' to quit): ")
      if my_input == "end":
          break
      # elif my_input.startswith("MCQ: "):
      #     cls.sample_choice(my_input)
      
      response = cls.sample_text(my_input)
      
      print(response) 