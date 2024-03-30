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

    eval_file: Optional[str] = field(default="", metadata={"help": "验证集"})
    task_type: str = field(
        default="sft", metadata={"help": "预训练任务：[sft, pretrain]"}
    )
    tokenize_num_workers: int = field(default=1, metadata={"help": ""})