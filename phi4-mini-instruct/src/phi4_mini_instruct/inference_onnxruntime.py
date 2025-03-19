import os
import onnxruntime_genai as og
import argparse
import time

model_id = os.environ['MODEL_ID']
hf_home = os.environ['HF_NOCACHE_HOME']
model_id_transformed = model_id.replace("/", "--")
onnx_model_path = f"{hf_home}/{model_id_transformed}/model.onnx"
model_home_dir = f"{hf_home}/{model_id_transformed}"

# https://github.com/openvinotoolkit/openvino/pull/29234
def main(args):
    if args.verbose: print("Loading model...")
    if args.timings:
        started_timestamp = 0
        first_token_timestamp = 0

    #config = og.Config(args.model_path)
    config = og.Config(model_home_dir)
    config.clear_providers()
    #if args.execution_provider != "cpu":
    #    if args.verbose: print(f"Setting model to {args.execution_provider}")
    #    config.append_provider(args.execution_provider)
    #else:
    # config.append_provider("cpu")        
    model = og.Model(config)

    if args.verbose: print("Model loaded")
    
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    if args.verbose: print("Tokenizer created")
    if args.verbose: print()
    search_options = {name:getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}
    
    # Set the max length to something sensible by default, unless it is specified by the user,
    # since otherwise it will be set to the entire context length
    if 'max_length' not in search_options:
        search_options['max_length'] = 2048

    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)
    print(model)

    # Keep asking for input prompts in a loop
    while True:
        text = input("Input: ")
        if not text:
            print("Error, input cannot be empty")
            continue

        if args.timings: started_timestamp = time.time()

        # If there is a chat template, use it
        prompt = f'{chat_template.format(input=text)}'

        input_tokens = tokenizer.encode(prompt)

        generator.append_tokens(input_tokens)
        if args.verbose: print("Generator created")

        if args.verbose: print("Running generation loop ...")
        if args.timings:
            first = True
            new_tokens = []

        print()
        print("Output: ", end='', flush=True)

        try:
            while not generator.is_done():
                generator.generate_next_token()
                if args.timings:
                    if first:
                        first_token_timestamp = time.time()
                        first = False

                new_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(new_token), end='', flush=True)
                if args.timings: new_tokens.append(new_token)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()

        if args.timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print(f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")
    parser.add_argument('-m', '--model_path', type=str, required=False, help='Onnx model folder path (must contain genai_config.json and model.onnx)')
    parser.add_argument('-e', '--execution_provider', type=str, required=False, choices=["cpu", "cuda", "dml"], help="Execution provider to run ONNX model with")
    parser.add_argument('-i', '--min_length', type=int, help='Min number of tokens to generate including the prompt')
    parser.add_argument('-l', '--max_length', type=int, help='Max number of tokens to generate including the prompt')
    parser.add_argument('-ds', '--do_sample', action='store_true', default=False, help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
    parser.add_argument('-p', '--top_p', type=float, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
    parser.add_argument('-r', '--repetition_penalty', type=float, help='Repetition penalty to sample with')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output and timing information. Defaults to false')
    parser.add_argument('-g', '--timings', action='store_true', default=False, help='Print timing information for each generation step. Defaults to false')
    args = parser.parse_args()
    main(args)
