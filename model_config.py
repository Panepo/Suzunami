from dotenv import load_dotenv
import os

load_dotenv()
model = os.getenv("MODEL")

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

DEFAULT_RAG_PROMPT = """\
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
"""

def phi_completion_to_prompt(completion):
    return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"


def llama3_completion_to_prompt(completion):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

if model == "llama3.2":
  model_configuration = {
    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    "start_message": DEFAULT_SYSTEM_PROMPT,
    "stop_tokens": ["<|eot_id|>"],
    "has_chat_template": True,
    "start_message": " <|start_header_id|>system<|end_header_id|>\n\n" + DEFAULT_SYSTEM_PROMPT + "<|eot_id|>",
    "history_template": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
    "current_message_template": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}",
    "console_message_start": "<|start_header_id|>user<|end_header_id|>",
    "console_message_end": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "rag_prompt_template": f"<|start_header_id|>system<|end_header_id|>\n\n{DEFAULT_RAG_PROMPT}<|eot_id|>"
    + """<|start_header_id|>user<|end_header_id|>


    Question: {input}
    Context: {context}
    Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>


    """,
    "completion_to_prompt": llama3_completion_to_prompt,
  }
elif model == "phi3":
  model_configuration = {
    "model_id": "microsoft/Phi-3-mini-4k-instruct",
    "remote_code": True,
    "start_message": "<|system|>\n{DEFAULT_SYSTEM_PROMPT}<|end|>\n",
    "history_template": "<|user|>\n{user}<|end|> \n<|assistant|>\n{assistant}<|end|>\n",
    "current_message_template": "<|user|>\n{user}<|end|> \n<|assistant|>\n{assistant}",
    "stop_tokens": ["<|end|>"],
    "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }<|end|>"""
    + """
    <|user|>
    Question: {input}
    Context: {context}
    Answer: <|end|>
    <|assistant|>""",
    "console_message_start": "<|user|>\n",
    "console_message_end": "<|end|> \n<|assistant|>\n",
    "completion_to_prompt": phi_completion_to_prompt,
  }
elif model == "tinyllama":
  model_configuration = {
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "remote_code": False,
    "start_message": f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}</s>\n",
    "history_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}</s> \n",
    "current_message_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}",
    "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }</s>"""
    + """
    <|user|>
    Question: {input}
    Context: {context}
    Answer: </s>
    <|assistant|>""",
  }
elif model == "phi3.5":
  model_configuration = {
    "model_id": "microsoft/Phi-3.5-mini-instruct",
    "remote_code": True,
    "start_message": "<|system|>\n{DEFAULT_SYSTEM_PROMPT}<|end|>\n",
    "history_template": "<|user|>\n{user}<|end|> \n<|assistant|>\n{assistant}<|end|>\n",
    "current_message_template": "<|user|>\n{user}<|end|> \n<|assistant|>\n{assistant}",
    "stop_tokens": ["<|end|>"],
    "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }<|end|>"""
    + """
    <|user|>
    Question: {input}
    Context: {context}
    Answer: <|end|>
    <|assistant|>""",
    "console_message_start": "<|user|>\n",
    "console_message_end": "<|end|> \n<|assistant|>\n",
    "completion_to_prompt": phi_completion_to_prompt,
  }
else:
  raise ValueError(f"Unknown depth: {model}")
