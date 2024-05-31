from transformers import AutoTokenizer, AutoConfig, AddedToken, LlamaTokenizer
import torch
from loguru import logger
import copy
import sys

from pathlib import Path
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from model.modeling_llama import LlamaForCausalLM


from template import get_template_and_fix_tokenizer
from peft import PeftModel
from transformers import GenerationConfig

def main():
    # 使用合并后的模型进行推理
    # model_name_or_path = 'Qwen/Qwen-7B-Chat'
    # template_name = 'qwen'
    #  adapter_name_or_path = None

    model_name_or_path = "/ML-A100/public/tmp/yiguofeng/contribute/Yi-mini-LLM/output/yi-1.1b-pretrain-v1/checkpoint-50000"
    template_name = "yi"
    adapter_name_or_path = "/ML-A100/public/tmp/yiguofeng/contribute/Yi-mini-LLM/output/yi-1.1b-ck50000-sft-v2"

    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = 100
    repetition_penalty = 1.3
    # 加载模型
    logger.info(f"Loading model from: {model_name_or_path}")
    logger.info(f"adapter_name_or_path: {adapter_name_or_path}")
    model = (
        LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto")
        .cuda()
        .eval()
    )

    # 加载adapter
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)

    tokenizer:LlamaTokenizer = LlamaTokenizer.from_pretrained(
        model_name_or_path if adapter_name_or_path is None else adapter_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    
    template = get_template_and_fix_tokenizer(tokenizer, "default")
    
    history = []

    query = input("User：")
    while True:
        query = query.strip()
        messages = [
            {"role": "user", "content": query}
        ]
        
        paired_messages = messages + [{"role": "assistant", "content": ""}]
        prompt_ids, _ = template.encode_oneturn(
            tokenizer=tokenizer, messages=paired_messages
        )
        
        prompt_length = len(prompt_ids)
        input_ids = torch.tensor([prompt_ids], device=model.device)
        
        print(f"input_ids: {input_ids}")
        decode_input = tokenizer.decode(
            input_ids[0]
        )
        print(f"decode_input: {decode_input}")
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
        )

        response_ids = output_ids[0][input_ids.shape[1]:]
        response = tokenizer.decode(
           response_ids, skip_special_tokens=True
        )
        

        print("Firefly：{}".format(response))
        query = input("User：")


if __name__ == "__main__":
    main()
