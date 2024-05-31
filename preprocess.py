import os
import json
import inspect
from itertools import chain
from functools import partial
from typing import Any, Callable, Dict, List, Union, Optional, Literal
from dataclasses import dataclass

from transformers import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer
from argument import CustomizedArguments
from template import Template, get_template_and_fix_tokenizer
from datasets import load_dataset, load_from_disk, Dataset, IterableDataset, Features
from loguru import logger
from utils import checksum, Role, merge_dataset, has_tokenized_data


DATA_CONFIG = "dataset_info.json"

IGNORE_INDEX = -100

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}


def best_fit_decreasing(strings, max_seq_len):
    # Step 1: Sort strings by length in decreasing order
    strings.sort(key=len, reverse=True)
    
    # Step 2: Prepare to store the result
    result = []
    used = [False] * len(strings)
    
    # Step 3: Try to fit smaller strings into the space remaining in larger strings
    for i in range(len(strings)):
        if not used[i]:
            # Current string as the base
            current = strings[i]
            used[i] = True
            # Try to append other strings to it
            for j in range(i + 1, len(strings)):
                if not used[j] and len(current) + len(strings[j]) <= max_seq_len:
                    current += strings[j]
                    used[j] = True
            # Append the combined string to the result list
            result.append(current)
    
    return result

def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    data_args: "CustomizedArguments",
) -> Dict[str, List[List[int]]]:
    # if data_args.use_best_fit_pack:
    #     text_examples = [
    #         messages[0]["content"] + tokenizer.eos_token
    #         for messages in examples["prompt"]
    #     ]
    #     block_size = data_args.max_seq_length
    #     split_texts = [
    #         text[i : i + block_size]
    #         for text in text_examples
    #         for i in range(0, len(text), block_size)
    #     ]
    #     best_fit_pack = best_fit_decreasing(split_texts, block_size)

    #     for idx, value in enumerate(best_fit_pack):
    #         result = {idx : value}
        
    #     return result
    # else:  # concatenation
        # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
    text_examples = [
        messages[0]["content"] + tokenizer.eos_token
        for messages in examples["prompt"]
    ]

    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
    concatenated_examples = {
        k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()
    }
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    block_size = data_args.max_seq_length
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    # we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # split by chunks of max_seq_length
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
    }

    return result


"""
https://github.com/hiyouga/LLaMA-Factory/tree/main/data
"""


def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "CustomizedArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    # https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/F.%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.md
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            continue

        messages = examples["prompt"][i] + examples["response"][i]
        input_ids, labels = [], []
        for turn_idx, (source_ids, target_ids) in enumerate(
            template.encode_multiturn(
                tokenizer,
                messages,
                examples["system"][i],
                examples["tools"][i],
                data_args.max_seq_length,
            )
        ):
            if turn_idx != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (
                    len(source_ids) - 1
                )
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            # source_ids即用户输入部分，target_ids即模型真实输出
            # 将用户输入的部分（问题）替换成了-100，保留了模型输入部分。在模型进行运算时，会根据input_ids的前面的tokens去预测下一个token
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs


def print_supervised_dataset_example(
    example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer"
) -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print(
        "inputs:\n{}".format(
            tokenizer.decode(example["input_ids"], skip_special_tokens=False)
        )
    )
    print("label_ids:\n{}".format(example["labels"]))
    print(
        "labels:\n{}".format(
            tokenizer.decode(
                list(filter(lambda x: x != IGNORE_INDEX, example["labels"])),
                skip_special_tokens=False,
            )
        )
    )


def print_unsupervised_dataset_example(
    example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer"
) -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print(
        "inputs:\n{}".format(
            tokenizer.decode(example["input_ids"], skip_special_tokens=False)
        )
    )


def get_preprocess_and_print_func(
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    args: "CustomizedArguments",
) -> Callable:
    if args.task_type == "pretrain":
        preprocess_func = partial(
            preprocess_pretrain_dataset,
            tokenizer=tokenizer,
            data_args=args,
        )
        print_function = partial(
            print_unsupervised_dataset_example, tokenizer=tokenizer
        )
    elif args.task_type == "sft":
        preprocess_func = partial(
            preprocess_supervised_dataset,
            tokenizer=tokenizer,
            template=template,
            data_args=args,
        )
        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)

    return preprocess_func, print_function


@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """

    """ basic configs """
    load_from: Literal["hf_hub", "ms_hub", "script", "file"]
    dataset_name: str
    """ extra configs """
    file_sha1: Optional[str] = None
    subset: Optional[str] = None
    folder: Optional[str] = None
    ranking: bool = False
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    """ columns """
    system: Optional[str] = None
    """ columns for the alpaca format """
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    """ columns for the sharegpt format """
    messages: Optional[str] = "conversations"
    tools: Optional[str] = None
    """ tags for the sharegpt format """
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(
        self, key: str, obj: Dict[str, Any], default: Optional[Any] = None
    ) -> None:
        setattr(self, key, obj.get(key, default))


def get_dataset_list(data_args: "CustomizedArguments") -> List["DatasetAttr"]:
    if data_args.dataset is not None:
        dataset_names = [ds.strip() for ds in data_args.dataset.split(",")]
    else:
        dataset_names = []

    try:
        with open(os.path.join(data_args.dataset_dir, DATA_CONFIG), "r") as f:
            dataset_info = json.load(f)
    except Exception as err:
        if len(dataset_names) != 0:
            raise ValueError(
                "Cannot open {} due to {}.".format(
                    os.path.join(data_args.dataset_dir, DATA_CONFIG), str(err)
                )
            )
        dataset_info = None

    if data_args.interleave_probs is not None:
        data_args.interleave_probs = [
            float(prob.strip()) for prob in data_args.interleave_probs.split(",")
        ]

    dataset_list: List[DatasetAttr] = []
    for name in dataset_names:

        if name not in dataset_info:
            raise ValueError("Undefined dataset {} in {}.".format(name, DATA_CONFIG))

        # 设置当前数据集的属性
        dataset_attr = DatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

        dataset_attr.set_attr("file_sha1", dataset_info[name])
        dataset_attr.set_attr("subset", dataset_info[name])
        dataset_attr.set_attr("folder", dataset_info[name])
        dataset_attr.set_attr("ranking", dataset_info[name], default=False)
        dataset_attr.set_attr("formatting", dataset_info[name], default="alpaca")

        if "columns" in dataset_info[name]:
            column_names = ["system"]
            if dataset_attr.formatting == "alpaca":
                column_names.extend(["prompt", "query", "response", "history"])
            else:  # 当前只支持sharegpt格式的agent tuning
                column_names.extend(["messages", "tools"])

            for column_name in column_names:
                dataset_attr.set_attr(column_name, dataset_info[name]["columns"])

        if dataset_attr.formatting == "sharegpt" and "tags" in dataset_info[name]:
            tag_names = (
                "role_tag",
                "content_tag",
                "user_tag",
                "assistant_tag",
                "observation_tag",
                "function_tag",
                "system_tag",
            )
            for tag in tag_names:
                dataset_attr.set_attr(tag, dataset_info[name]["tags"])

        dataset_list.append(dataset_attr)

    return dataset_list


# @TODO 需要改进
def load_single_dataset(
    dataset_attr: "DatasetAttr",
    data_args: "CustomizedArguments",
) -> Union["Dataset", "IterableDataset"]:
    logger.info("Loading dataset {}...".format(dataset_attr))
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                if FILEEXT2TYPE.get(file_name.split(".")[-1], None) != None:
                    data_files.append(os.path.join(local_path, file_name))
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                    raise ValueError("File types should be identical.")
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        else:
            raise ValueError("File not found.")

        if data_path is None:
            raise ValueError("File extension must be txt, csv, json or jsonl.")

        checksum(data_files, dataset_attr.file_sha1)
    else:
        raise NotImplementedError

    if (
        "trust_remote_code" in inspect.signature(load_dataset).parameters
    ):  # for datasets==2.16.0
        kwargs = {"trust_remote_code": True}
    else:
        kwargs = {}

    try:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=data_args.split,
            cache_dir=data_args.cache_dir,
            **kwargs,
        )
    except Exception as e:
        raise ValueError("Error load dataset {}.".format(e))


    if data_args.max_samples is not None:  # truncate dataset
        num_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(num_samples))

    return align_dataset(dataset, dataset_attr, data_args)


def convert_alpaca(
    examples: Dict[str, List[Any]], dataset_attr: "DatasetAttr"
) -> Dict[str, List[Any]]:
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    for i in range(len(examples[dataset_attr.prompt])):
        prompt = []
        if dataset_attr.history and isinstance(examples[dataset_attr.history][i], list):
            for old_prompt, old_response in examples[dataset_attr.history][i]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        content = []
        if dataset_attr.prompt and examples[dataset_attr.prompt][i]:
            content.append(examples[dataset_attr.prompt][i])

        if dataset_attr.query and examples[dataset_attr.query][i]:
            content.append(examples[dataset_attr.query][i])

        prompt.append({"role": Role.USER.value, "content": "\n".join(content)})

        if dataset_attr.response and isinstance(
            examples[dataset_attr.response][i], list
        ):
            response = [
                {"role": Role.ASSISTANT.value, "content": content}
                for content in examples[dataset_attr.response][i]
            ]
        elif dataset_attr.response and isinstance(
            examples[dataset_attr.response][i], str
        ):
            response = [
                {
                    "role": Role.ASSISTANT.value,
                    "content": examples[dataset_attr.response][i],
                }
            ]
        else:
            response = []

        # The above code is using Python to output three hash symbols "
        outputs["prompt"].append(prompt)
        outputs["response"].append(response)
        outputs["system"].append(
            examples[dataset_attr.system][i] if dataset_attr.system else ""
        )
        outputs["tools"].append("")

    return outputs


def convert_sharegpt(
    examples: Dict[str, List[Any]], dataset_attr: "DatasetAttr"
) -> Dict[str, List[Any]]:
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
    }
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
    accept_tags = (odd_tags, even_tags)
    for i, messages in enumerate(examples[dataset_attr.messages]):
        if (
            dataset_attr.system_tag
            and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag
        ):
            system = messages[0][dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = examples[dataset_attr.system][i] if dataset_attr.system else ""

        messages = messages[: len(messages) // 2 * 2]  # should be multiples of 2
        if len(messages) == 0:
            continue

        aligned_messages = []
        for turn_idx, message in enumerate(messages):
            if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                raise ValueError("Invalid role tag in {}.".format(messages))

            aligned_messages.append(
                {
                    "role": tag_mapping[message[dataset_attr.role_tag]],
                    "content": message[dataset_attr.content_tag],
                }
            )

        outputs["prompt"].append(aligned_messages[:-1])
        outputs["response"].append(aligned_messages[-1:])
        outputs["system"].append(system)
        outputs["tools"].append(
            examples[dataset_attr.tools][i] if dataset_attr.tools else ""
        )

    return outputs


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "CustomizedArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        system: "..."
        tools: "..."
    """
    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca, dataset_attr=dataset_attr)
    else:
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr)

    column_names = list(next(iter(dataset)).keys())
    features = Features.from_dict(
        {
            "prompt": [
                {
                    "role": {"dtype": "string", "_type": "Value"},
                    "content": {"dtype": "string", "_type": "Value"},
                }
            ],
            "response": [
                {
                    "role": {"dtype": "string", "_type": "Value"},
                    "content": {"dtype": "string", "_type": "Value"},
                }
            ],
            "system": {"dtype": "string", "_type": "Value"},
            "tools": {"dtype": "string", "_type": "Value"},
        }
    )

    kwargs = dict(
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache),
        desc="Converting format of dataset",
    )

    return dataset.map(
        convert_func,
        batched=True,
        remove_columns=column_names,
        features=features,
        **kwargs,
    )


def get_dataset(
    tokenizer: "PreTrainedTokenizer",
    args: "CustomizedArguments",
    training_args: "TrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    template = get_template_and_fix_tokenizer(tokenizer, args.template_name)
    logger.info(
        f"tokenizer eos token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}"
    )
    # Load tokenized dataset
    if args.tokenized_path is not None:
        if has_tokenized_data(args.tokenized_path):
            logger.warning(
                "Loading dataset from disk will ignore other data arguments."
            )
            dataset = load_from_disk(args.tokenized_path)
            logger.info("Loaded tokenized dataset from {}.".format(args.tokenized_path))

            return dataset

    with training_args.main_process_first(desc="load dataset"):
        all_datasets = []
        for dataset_attr in get_dataset_list(data_args=args):
            all_datasets.append(load_single_dataset(dataset_attr, args))
        dataset = merge_dataset(all_datasets, args, training_args)

    with training_args.main_process_first(desc="pre-process dataset"):
        preprocess_func, print_function = get_preprocess_and_print_func(
            tokenizer, template, args
        )
        column_names = list(next(iter(dataset)).keys())
        kwargs = dict(
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=(not args.overwrite_cache),
            desc="Running tokenizer on dataset",
        )

        dataset = dataset.map(
            preprocess_func, batched=True, remove_columns=column_names, **kwargs
        )
        logger.info(f"Total training number: {len(dataset)}")

        if args.task_type == "pretrain":
            logger.info(
                f"Total training tokens: {len(dataset) * args.max_seq_length // 1e9} B Tokens"
            )
        else:
            logger.info(
                f"Total training tokens: {len(dataset) * args.max_seq_length // 1e3} K Tokens"
            )

        if args.tokenized_path is not None:
            if training_args.should_save:
                dataset.save_to_disk(args.tokenized_path)
                logger.info(
                    "Tokenized dataset saved at {}.".format(args.tokenized_path)
                )
                logger.info(
                    "Please restart the training with `--tokenized_path {}`.".format(
                        args.tokenized_path
                    )
                )

            exit(0)

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                raise RuntimeError(
                    "Cannot find valid samples, check `data/README.md` for the data format."
                )

        return dataset
