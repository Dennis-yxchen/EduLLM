from typing import override
from openai import OpenAI


import os
import openai
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.language_model.base_gpt_model import BaseGPTModel
from concordia.document import interactive_document
from collections.abc import Collection, Sequence

class CustomBaseGPTModel(BaseGPTModel):
    @override
    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        # Limit tokens to 4000 for GPT models
        max_tokens = min(max_tokens, 4000)

        if system_prompt is None:
            messages = [
                {'role': 'system',
                'content': ('You are a helpful assistant. ')},
                {'role': 'user',
                'content': prompt}
            ]
        else:
            messages = [
                {'role': 'system',
                'content': system_prompt},
                {'role': 'user',
                'content': prompt}
            ]

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            stop=terminators,
            seed=seed,
        )

        if self._measurements is not None:
            self._measurements.publish_datum(
                self._channel,
                {'raw_text_length': len(response.choices[0].message.content)},
            )
        return response.choices[0].message.content


class CustomOpenAI(CustomBaseGPTModel):
  """Language Model that uses OpenAI GPT models."""
  def __init__(
      self,
      model_name: str,
      *,
      api_key: str | None = None,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ):
    """Initializes the instance.

    Args:
      model_name: The language model to use. For more details, see
        https://platform.openai.com/docs/guides/text-generation/which-model-should-i-use.
      api_key: The API key to use when accessing the OpenAI API. If None, will
        use the OPENAI_API_KEY environment variable.
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
    """
    if api_key is None:
        api_key = os.environ['OPENAI_API_KEY']
    self._api_key = api_key
    client = OpenAI(
        base_url='https://api.siliconflow.cn/v1',
        api_key=self._api_key,
    )
    
    super().__init__(model_name=model_name,
                     client=client,
                     measurements=measurements,
                     channel=channel)

def get_model(model_name: str, api_key: str) -> CustomOpenAI:
    model = CustomOpenAI(
        model_name=model_name,
        api_key=api_key,
        )
    
    return model


if __name__ == "__main__":
    model = CustomOpenAI(
        model_name="deepseek-ai/DeepSeek-V3",
        api_key="sk-ufvfjzrydqzznjfnqabneayuhyimirhnwekmiemjyskvxedo",
        )
    prompt = interactive_document.InteractiveDocument(model)
    ans = prompt.open_question(
        "Who are you",
        terminators=(),
        max_tokens=4096,
    )
    print(ans)
    
    
    
