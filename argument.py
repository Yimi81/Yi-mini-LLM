from dataclasses import dataclass, field
from typing import Optional, Literal
from transformers import Seq2SeqTrainingArguments


@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """

    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    max_seq_length: int = field(default=512, metadata={"help": "输入最大长度"})
    template_name: str = field(default=None, metadata={"help": "sft时的数据格式"})
    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."
        },
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models and datasets downloaded from huggingface.co or modelscope.cn."},
    )
    split: str = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."},
    )
    mix_strategy: Literal["concat", "interleave_under", "interleave_over"] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."},
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={
            "help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    tokenized_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save or load the tokenized datasets."},
    )
    eval_file: Optional[str] = field(default="", metadata={"help": "验证集"})
    task_type: str = field(
        default="sft", metadata={"help": "预训练任务：[sft, pretrain]"}
    )
    train_mode: str = field(
        default="lora", metadata={"help": "训练方式：[full, lora, dora]"}
    )

    flash_attn: bool = field(default=True)
    use_dora: bool = field(default=False)
    lora_rank: Optional[int] = field(
        default=8, metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={
            "help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."
        },
    )
    lora_dropout: Optional[float] = field(
        default=0.0, metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )

    use_best_fit_pack: bool = field(default=False)
    