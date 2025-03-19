#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import openvino_genai

model_id = os.environ.get('MODEL_ID', None)
ov_home = os.environ.get('OV_HOME', None)

if not model_id or not ov_home:
    raise EnvironmentError("Please set environment variables: MODEL_ID and OV_HOME.")

model_id_transformed = model_id.replace("/", "--")
model_home_dir = os.path.join(ov_home, model_id_transformed)

def streamer(subword):
    print(subword, end='', flush=True)
    return False


def infer(args):
    device = 'CPU'  # GPU can be used as well.
    pipe = openvino_genai.LLMPipeline(model_home_dir, device)

    config = build_generation_config(
        pipe,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        perf_text="",
        verbose=args.verbose,
        timings=args.timings
    )

    pipe.start_chat()
    while True:
        try:
            prompt = input('question:\n')
        except EOFError:
            break
        pipe.generate(prompt, config, streamer)
        print('\n----------')
    pipe.finish_chat()


def build_generation_config(
    pipe,
    top_p: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 0,
    max_new_tokens: int = 50,
    perf_text: str = "",
    verbose: bool = False,
    timings: bool = False
):
    """
    Generates text from a user-supplied prompt, using sampling (not beam search).

    Setting generation_type = GenerationType.SAMPLING ensures no "beam_idx" port is required.

    Args:
        user_text (str): The user's prompt.
        top_p (float): Nucleus (top-p) sampling threshold.
        temperature (float): Controls randomness of the output.
        top_k (int): Restricts sampling to the top k tokens.
        max_new_tokens (int): Maximum number of tokens to generate.
        perf_text (str): For performance messages (optional).
        verbose (bool): If True, prints debug information.
        timings (bool): If True, prints duration of generation.

    Returns:
        model_output (str): The complete generated text.
        perf_text (str): Performance string (if timings is True).
    """
    # Get the config from the pipeline
    config = pipe.get_generation_config()


    #config.temperature = temperature
    #config.top_p = top_p
    #if top_k > 0:
    #    config.top_k = top_k
    #config.max_new_tokens = max_new_tokens
    config.max_length = 2048

    print(f"{config}")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="OpenVINO + openvino_genai text generation with sampling only."
    )
    parser.add_argument('-t', '--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('-p', '--top_p', type=float, default=0.9,
                        help='Top p (nucleus) sampling threshold')
    parser.add_argument('-k', '--top_k', type=int, default=50,
                        help='Top k tokens to sample from')
    parser.add_argument('-l', '--max_new_tokens', type=int, default=100,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('-g', '--timings', action='store_true',
                        help='Print timing information')
    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()
