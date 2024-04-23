import os
import json
import torch
from argparse import ArgumentParser
from loguru import logger
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    LlamaTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
)
from model.configuration_llama import LlamaConfig
from model.modeling_llama import LlamaForCausalLM

import datasets
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from preprocess import get_dataset, IGNORE_INDEX
from argument import CustomizedArguments
from peft import LoraConfig, get_peft_model, TaskType
from template import get_template_and_fix_tokenizer


def setup_everything():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_args_file", type=str, default="hparams/train_args.json"
    )
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    # 解析得到自定义参数和Trainer自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(os.path.join(training_args.output_dir, "train.log"))
    logger.info("train_args:{}".format(training_args))
    # 加载训练配置文件
    with open(train_args_file, "r") as f:
        train_args = json.load(f)
    # 保存训练参数到输出目录
    with open(os.path.join(training_args.output_dir, "train_args.json"), "w") as f:
        json.dump(train_args, f, indent=4)
    # 设置随机种子
    set_seed(training_args.seed)

    # check some setting
    assert args.task_type in [
        "pretrain",
        "sft",
    ], "task_type should be in ['pretrain', 'sft']"
    assert args.train_mode in [
        "full",
        "lora",
        "dora",
    ], "train_mode should be in ['full', 'lora', 'dora']"
    assert (
        sum([training_args.fp16, training_args.bf16]) == 1
    ), "only one of fp16 and bf16 can be True"

    return args, training_args


def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层, 为所有全连接添加adapter
    """
    assert train_mode in ["lora", "dora"]
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    lora_module_names = list(lora_module_names)
    logger.info(f"LoRA target module names: {lora_module_names}")
    return lora_module_names


def load_model(args, training_args):
    """
    加载模型
    """
    assert training_args.bf16 or training_args.fp16, "bf16 or fp16 should be True"
    logger.info(f"Loading model from base model: {args.model_name_or_path}")
    logger.info(f"Train model with {args.train_mode}")

    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    kwargs = dict(torch_dtype=torch_dtype)
    if args.flash_attn:
        kwargs.update(attn_implementation="flash_attention_2")
    config = LlamaConfig.from_pretrained("./model")
    config.update(kwargs)
    logger.info(f"model config {config}")
    model = LlamaForCausalLM(config=config)
    if args.train_mode == "lora" and args.task_type in ["pretrain", "sft"]:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # init peft_config
    if args.train_mode == "full":
        peft_config = None
    else:
        # 目前默认lora target是所有全连接层
        target_modules = find_all_linear_names(model, args.train_mode)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            use_dora=args.use_dora,
        )

    # init peft model
    if args.train_mode in ["lora", "dora"] and args.task_type in ["pretrain", "sft"]:
        model = get_peft_model(model, peft_config)
        logger.info(
            f"memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB"
        )
        model.print_trainable_parameters()

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return {"model": model, "peft_config": peft_config}


def init_components(args, training_args):
    """
    初始化各个组件
    """
    training_args.ddp_find_unused_parameters = False
    logger.info("Initializing components...")

    # 加载tokenzier
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )
    logger.info(f"vocab_size of tokenizer: {tokenizer.vocab_size}")

    # 加载模型
    components = load_model(args, training_args)
    model = components['model']

    # 初始化dataset和collator
    if args.task_type == "pretrain":
        logger.info("Train model with pretrain task")
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    elif args.task_type == "sft":
        logger.info("Train model with sft task")
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX
        )

    train_dataset = get_dataset(tokenizer, args, training_args)

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
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
