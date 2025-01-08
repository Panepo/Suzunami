from dotenv import load_dotenv
import os

load_dotenv()
device = os.getenv("DEVICE")
model = os.getenv("MODEL")

if device in ["cpu", "cuda", "gpu", "directml"]:
  pass
else:
  raise ValueError(f"Unknown device: {device}")

from model_download import llama32_dir, phi35_dir, phi3_dir

if model == "llama3.2":
  if device == "cuda":
    model_dir = str(llama32_dir) + r"\cuda\cuda-int4-rtn-block-32"
  else:
    model_dir = str(llama32_dir) + r"\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4"
elif model == "phi3.5":
  if device == "gpu":
    model_dir = str(phi35_dir) + r"\gpu\gpu-int4-awq-block-128"
  else:
    model_dir = str(phi35_dir) + r"\cpu_and_mobile\cpu-int4-awq-block-128-acc-level-4"
elif model == "phi3":
  if device == "cuda":
    model_dir = str(phi3_dir) + r"\cuda\cuda-int4-rtn-block-128"
  elif device == "directml":
    model_dir = str(phi3_dir) + r"\directml\directml-int4-rtn-block-128"
  else:
    model_dir = str(phi3_dir) + r"\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4"
else:
  raise ValueError(f"Unknown model: {model}")

import onnxruntime_genai as og

model = og.Model(model_dir)
tokenizer = og.Tokenizer(model)

# Set the max length to something sensible by default,
# since otherwise it will be set to the entire context length
search_options = {}
search_options['max_length'] = 2048
search_options['batch_size'] = 1
