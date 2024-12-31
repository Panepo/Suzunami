import time

def demo():
  started_timestamp = time.time()
  first = True
  new_tokens = []

  import onnxruntime_genai as og
  from llm import model, tokenizer, search_options
  from model_config import model_configuration, DEFAULT_SYSTEM_PROMPT

  start_message = model_configuration.get("start_message", DEFAULT_SYSTEM_PROMPT)
  console_message_start = model_configuration.get("console_message_start", "")
  console_message_end = model_configuration.get("console_message_end", "")
  tokenizer_stream = tokenizer.create_stream()

  params = og.GeneratorParams(model)

  while 1:
    print("================================================")
    message = input("Please say something: ")
    if message == "exit":
      break

    prompt = start_message + console_message_start + message + console_message_end
    input_tokens = tokenizer.encode(prompt)
    params.set_search_options(**search_options)

    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    print("================================================")
    print("Output: ", end='', flush=True)

    try:
      while not generator.is_done():
        generator.generate_next_token()
        if first:
          first = False
          first_token_timestamp = time.time()

        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end='', flush=True)
        new_tokens.append(new_token)
    except KeyboardInterrupt:
        print("  --control+c pressed, aborting generation--")
        break

    print("================================================")
    prompt_time = first_token_timestamp - started_timestamp
    run_time = time.time() - first_token_timestamp
    print(f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps")




if __name__ == "__main__":
  demo()
