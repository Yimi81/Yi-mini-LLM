import os
from argparse import ArgumentParser
from loguru import logger
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    LlamaTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
)
from model.configuration_llama import LlamaConfig
from model.modeling_llama import LlamaForCausalLM

import datasets
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from preprocess import get_preprocess_and_print_func
from argument import CustomizedArguments

attn_implementation = 'flash_attention_2'
try:
    from flash_attn import flash_attn_func
except Exception as e:
    attn_implementation = 'eager'

def setup_everything():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_args_file", type=str, default="hparams/train_args.json"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    # 读取训练的参数配置
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    # 解析得到自定义参数和Trainer自带参数
    args, training_args = parser.parse_json_file(json_file=args.train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(os.path.join(training_args.output_dir, "train.log"))
    logger.info("train_args:{}".format(training_args))
    # 设置随机种子
    set_seed(training_args.seed)
    return args, training_args


def load_pretrain_dataset(args, training_args, tokenizer):
    """
    加载数据集
    """
    data_path = args.train_file
    cache_dir = os.path.join(os.path.dirname(data_path), ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 扫描所有json文件
    logger.info("Scanning all the pre-training file...")
    files = []
    for root, dir_names, file_names in os.walk(data_path):
        for file_name in file_names:
            file = os.path.join(root, file_name)
            if file_name.endswith(".jsonl") or file_name.endswith(".json"):
                files.append(file)
    logger.info(f"Total num of training file: {len(files)}")

    # 预处理所有来自不同源的文本，tokenize&packing
    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        pretrain_dataset = []  # 汇总所有dataset
        for idx, file in enumerate(tqdm(files)):
            logger.info(f"Loading file: {file}")
            file_name = os.path.basename(file)
            file_name = (
                file_name.replace(".jsonl", "")
                if "jsonl" in file_name
                else file_name.replace(".json", "")
            )
            cache_path = os.path.join(cache_dir, file_name)
            os.makedirs(cache_dir, exist_ok=True)

            try:
                # 已有缓存
                processed_dataset = datasets.load_from_disk(
                    cache_path, keep_in_memory=False
                )

                logger.info(f"Finished loading datasets-{file_name} from cache")
            except Exception:
                # 初次加载做处理并缓存
                tmp_cache_path = os.path.join(cache_path, "tmp")
                logger.info(
                    f"There is no cache of file {file_name}, start preprocessing..."
                )

                preprocess_func = None
                # 根据数据源获取不同预处理方法
                if "wikipedia" in file:
                    preprocess_func = get_preprocess_and_print_func(
                        tokenizer,
                        data_args=args,
                        data_source="wikipedia",
                    )
                elif "skypile" in file:
                    preprocess_func = get_preprocess_and_print_func(
                        tokenizer,
                        data_args=args,
                        data_source="skypile",
                    )

                dataset = load_dataset(
                    "json",
                    data_files=file,
                    cache_dir=tmp_cache_path,
                    keep_in_memory=False,
                )
                column_names = dataset.column_names["train"]

                kwargs = dict(
                    num_proc=args.tokenize_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={
                        k: os.path.join(tmp_cache_path, "tokenized.arrow")
                        for k in dataset
                    },
                    desc="Running tokenizer on dataset",
                )

                dataset = dataset.map(
                    preprocess_func,
                    batched=True,
                    remove_columns=column_names,
                    **kwargs,
                )

                processed_dataset = dataset
                processed_dataset.save_to_disk(cache_path)

            logger.info(
                f"Training number of {file_name}: {len(processed_dataset['train'])}"
            )
            if idx == 0:
                pretrain_dataset = processed_dataset["train"]
            else:
                assert (
                    pretrain_dataset.features.type
                    == processed_dataset["train"].features.type
                )
                pretrain_dataset = concatenate_datasets(
                    [pretrain_dataset, processed_dataset["train"]]
                )

    logger.info(f"Total training number: {len(pretrain_dataset)}")
    logger.info(
        f"Total training tokens: {len(pretrain_dataset) * args.max_seq_length // 1e9} B Tokens"
    )

    return pretrain_dataset


def init_components(args, training_args):
    """
    初始化各个组件
    """
    logger.info("Initializing components...")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    training_args.ddp_find_unused_parameters = False if ddp else None

    # 初始化模型
    kwargs = {}
    if args.flash_attn:
        kwargs = dict(attn_implementation="flash_attention_2")
    config = LlamaConfig.from_pretrained("./model", **kwargs)
    model = LlamaForCausalLM(config)

    # 加载tokenzier
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    # 预训练
    train_dataset = load_pretrain_dataset(args, training_args, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    return trainer


def main():
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = init_components(args, training_args)
    # 开始训练
    logger.info("*** starting training ***")
    train_result = trainer.train()
    # 保存最好的checkpoint
    final_save_path = os.path.join(training_args.output_dir)
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()
