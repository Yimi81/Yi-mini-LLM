from itertools import chain
from functools import partial
from typing import Any, Callable, Dict, List

from transformers.tokenization_utils import PreTrainedTokenizer
from argument import CustomizedArguments


def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    data_args: "CustomizedArguments",
    data_source: "str",
) -> Dict[str, List[List[int]]]:
    # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
    if data_source == "wikipedia": 
        text_examples = [messages + tokenizer.eos_token for messages in examples["completion"]]
    elif data_source == "skypile" or data_source == "map-cc": 
        text_examples = [messages + tokenizer.eos_token for messages in examples["text"]]

    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
    concatenated_examples = {
        k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()
    }
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    block_size = data_args.max_seq_length
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (total_length // block_size) * block_size
    # split by chunks of max_seq_length
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    return result


def get_preprocess_and_print_func(
    tokenizer: "PreTrainedTokenizer",
    data_args: "CustomizedArguments",
    data_source: "str",
) -> Callable:
    return partial(
        preprocess_pretrain_dataset,
        tokenizer=tokenizer,
        data_args=data_args,
        data_source=data_source,
    )
