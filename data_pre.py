from datasets import load_dataset

# 使用 split 参数分批加载数据
try:
    dataset = load_dataset('json', data_files='/ML-A100/public/tmp/yiguofeng/contribute/Yi-mini-LLM/data/matrix_book_format.jsonl', split='train[:10%]', cache_dir="/ML-A100/public/tmp/yiguofeng/contribute/Yi-mini-LLM/data/.cache")
except Exception as e:
    print(f"Error loading dataset: {e}")

print("load finish")