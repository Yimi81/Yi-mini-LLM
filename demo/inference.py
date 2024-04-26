import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def parse_inputs():
    parser = argparse.ArgumentParser(description="Mini-Yi inference demo")
    parser.add_argument(
        "--model",
        type=str,
        default="model-path",
        help="pretrained model path locally or name on huggingface",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="max number of tokens to generate",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="whether to enable streaming text generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="《小王子》是一本畅销童话书，它讲述了：",
        help="The prompt to start with",
    )
    args = parser.parse_args()
    return args


def main(args):
    print(args)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto"
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextStreamer(tokenizer) if args.streaming else None
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        streamer=streamer,
        do_sample=True,
        repetition_penalty=1.3
    )

    if streamer is None:
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    args = parse_inputs()
    main(args)
