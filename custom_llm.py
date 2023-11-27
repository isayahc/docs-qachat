from threading import Thread
from typing import Optional

import gradio as gr
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def init_chain(model, tokenizer, prompt):
    class CustomLLM(LLM):

        """Streamer Object"""

        streamer: Optional[TextIteratorStreamer] = None

        def _call(self, prompt, stop=None, run_manager=None) -> str:
            self.streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)
            inputs = tokenizer(prompt, return_tensors="pt")
            kwargs = dict(input_ids=inputs["input_ids"], streamer=self.streamer, max_new_tokens=20)
            thread = Thread(target=model.generate, kwargs=kwargs)
            thread.start()
            return ""

        @property
        def _llm_type(self) -> str:
            return "custom"

    llm = CustomLLM()


    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain, llm