from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """

    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    train_file: str = field(
        default="/ML-A100/public/tmp/yiguofeng/contribute/Yi-mini-LLM/data/pretrain",
        metadata={
            "help": "训练集。如果task_type=pretrain, 请指定文件夹, 将扫描其下面的所有json/jsonl文件"
        },
    )
    max_seq_length: int = field(default=512, metadata={"help": "输入最大长度"})
    template_name: str = field(default="", metadata={"help": "sft时的数据格式"})

    eval_file: Optional[str] = field(default="", metadata={"help": "验证集"})
    task_type: str = field(
        default="sft", metadata={"help": "预训练任务：[sft, pretrain]"}
    )
    train_mode: str = field(
        default="lora", metadata={"help": "训练方式：[full, lora, dora]"}
    )
    tokenize_num_workers: int = field(default=10, metadata={"help": "预训练时tokenize的线程数量"})

    flash_attn: bool = field(default=True)
    use_dora: bool = field(default=False)
    lora_rank: Optional[int] = field(default=8, metadata={"help": "The intrinsic dimension for LoRA fine-tuning."})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."})
    lora_dropout: Optional[float] = field(default=0.0, metadata={"help": "Dropout rate for the LoRA fine-tuning."})